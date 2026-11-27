from moviad.datasets.vad_dataset import VADDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.utilities.get_sizes import params_to_mb, count_params

from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

class PatchCoreCL(ContinualModel):

    def __init__(self, patchcore_model: PatchCore, max_train_patches: int = 4_000_000,
                 foreground_fn: Callable[[torch.Tensor], torch.Tensor] | None = None):
        super().__init__(patchcore_model)

        # change the patchcore memory bank with a set of memory banks, one per task
        self.vad_model.memory_bank = {}
        self.n_samples_per_task = self.vad_model.memory_bank_size
        # cap on the coreset candidate patches (kept in fp16 on CPU): with a fixed
        # fraction per image the RAM grows with the dataset (23k bees at
        # 512x512, layer1-3 → ~38M patches, ~270GB in fp32 + the torch.cat copy)
        self.max_train_patches = max_train_patches
        # foreground_fn(images) -> object mask (B,1,H,W) at the input
        # resolution. If given: during training patches are sampled from the
        # object first (the memory bank is not wasted on the background); at inference
        # the image score and the map ignore the background
        self.foreground_fn = foreground_fn

    def _grid_mask(self, images: torch.Tensor, pixel_mask: torch.Tensor, grid_hw: tuple) -> torch.Tensor:
        """Mask (B, h*w) on the feature grid: patches that touch the object."""
        grid = F.adaptive_max_pool2d(pixel_mask.float(), grid_hw)
        return grid.reshape(images.shape[0], -1).to(self.vad_model.device) > 0

    def _rebalance_memory_bank(self):
        n_tasks = len(self.vad_model.memory_bank) + 1
        self.n_samples_per_task = self.vad_model.memory_bank_size // n_tasks
        self.vad_model.coreset_extractor.k = self.n_samples_per_task

        for task_id in self.vad_model.memory_bank:
            print("Rebalancing Memory Bank for Task", task_id, "with new sample limit:", self.n_samples_per_task)
            embeddings = self.vad_model.memory_bank[task_id]
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings)
            self.vad_model.memory_bank[task_id] = coreset

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        self._rebalance_memory_bank()

    def train_task(self, task_index: int, train_dataset, eval_dataset, metrics, device, logger = None, train_args:TrainingArgs = None):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4,
            prefetch_factor=2
        )

        # same number of random patches for each image, within max_train_patches
        # in total, written to a preallocated fp16 buffer (no torch.cat)
        patches_per_image = max(1, self.max_train_patches // len(train_dataset))
        embeddings = None
        n_filled = 0

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                embedding, batch_size, grid_h, grid_w = self.vad_model.extract_embedding(images)
                embedding = embedding.reshape(batch_size, -1, embedding.shape[-1])
                num_patches = embedding.shape[1]
                keep = min(patches_per_image, num_patches)
                # random order; with foreground_fn the object's patches come first
                # (if the object has fewer than keep patches, the background fills the rest)
                priority = torch.rand(batch_size, num_patches, device=embedding.device)
                if self.foreground_fn is not None:
                    fg = self._grid_mask(images, self.foreground_fn(images.to(embedding.device)),
                                         (grid_h, grid_w))
                    priority = priority + fg.to(priority.device).float()
                indices = priority.argsort(dim=1, descending=True)[:, :keep]
                sampled = torch.gather(embedding, 1, indices.unsqueeze(-1).expand(-1, -1, embedding.shape[-1]))
                sampled = sampled.reshape(-1, embedding.shape[-1]).half().cpu()

                if embeddings is None:
                    embeddings = torch.empty((len(train_dataset) * keep, sampled.shape[1]), dtype=torch.float16)
                    print(f"  {keep} patches per image, buffer {tuple(embeddings.shape)} "
                          f"({embeddings.numel() * 2 / 1e9:.1f} GB fp16)")
                embeddings[n_filled:n_filled + sampled.shape[0]] = sampled
                n_filled += sampled.shape[0]

            embeddings = embeddings[:n_filled]
            torch.cuda.empty_cache()

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings).float()

            self.vad_model.memory_bank[task_index] = coreset


    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass

    def forward(self, batch: torch.Tensor):

      anomaly_maps, pred_scores, mean_distances = [], [], []
      embedding, batch_size, width, height = self.vad_model.extract_embedding(batch)
      image_size = batch.shape[2:]
      if self.foreground_fn is not None:
          pixel_mask = self.foreground_fn(batch.to(self.vad_model.device)).float()
          grid_mask = self._grid_mask(batch, pixel_mask, (width, height))

      for task_id in self.vad_model.memory_bank:
          task_memory_bank = self.vad_model.memory_bank[task_id].to(self.vad_model.device)

          # Compute NN distances once per task
          patch_scores, locations = self.vad_model.nearest_neighbors(
              embedding=embedding, n_neighbors=1, memory_bank=task_memory_bank
          )

          # Mean NN distance per sample — used for task identification (matches original criterion)
          mean_dist = patch_scores.reshape(batch_size, -1).mean(dim=1)  # (batch_size,)
          mean_distances.append(mean_dist)

          # Reshape for score and map computation
          patch_scores_2d = patch_scores.reshape(batch_size, -1)       # (batch, H*W)
          locations_2d   = locations.reshape(batch_size, -1)           # (batch, H*W)

          # Anomaly score (only on the object's patches, if there is a mask)
          score_patches = patch_scores_2d * grid_mask if self.foreground_fn is not None else patch_scores_2d
          pred_score = self.vad_model.compute_anomaly_score_old(
              score_patches, locations_2d, embedding, task_memory_bank
          )

          # Anomaly map, zeroed outside the object (crop border and halo on the background)
          patch_scores_spatial = patch_scores_2d.reshape(batch_size, 1, width, height)
          batch_anomaly_maps = self.vad_model.anomaly_map_generator(
              patch_scores_spatial, image_size=image_size
          )
          if self.foreground_fn is not None:
              batch_anomaly_maps = batch_anomaly_maps * pixel_mask

          anomaly_maps.append(batch_anomaly_maps)
          pred_scores.append(pred_score)

      anomaly_maps    = torch.stack(anomaly_maps,    dim=0)  # (n_tasks, batch, 1, H, W)
      anomaly_scores  = torch.stack(pred_scores,     dim=0)  # (n_tasks, batch)
      mean_distances  = torch.stack(mean_distances,  dim=0)  # (n_tasks, batch)

      mean_distances = mean_distances.T                       # (batch, n_tasks)
      anomaly_scores = anomaly_scores.T                       # (batch, n_tasks)
      anomaly_maps   = anomaly_maps.permute(1, 0, 2, 3, 4)   # (batch, n_tasks, 1, H, W)

      # Select the task with the minimum MEAN NN distance (not anomaly score)
      min_task_idx = mean_distances.argmin(dim=1)             # (batch,)
      batch_idx    = torch.arange(batch_size, device=batch.device)

      min_anomaly_maps = anomaly_maps[batch_idx, min_task_idx]   # (batch, 1, H, W)
      min_scores       = anomaly_scores[batch_idx, min_task_idx] # (batch,)

      return min_anomaly_maps, min_scores

    def get_model_size(self):
        feature_extractor_size = self.vad_model.feature_extractor.get_size()

        memory_bank_size = 0
        for task_id in self.vad_model.memory_bank:
            memory_bank_size += params_to_mb(count_params(self.vad_model.memory_bank[task_id]))

        return {
            "feature_extractor": feature_extractor_size,
            "memory_bank": memory_bank_size,
            "total": feature_extractor_size + memory_bank_size
        }
