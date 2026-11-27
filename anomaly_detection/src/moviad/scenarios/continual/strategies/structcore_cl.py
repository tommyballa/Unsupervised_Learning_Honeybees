from moviad.datasets.vad_dataset import VADDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.structcore.structcore import StructCore
from moviad.models.training_args import TrainingArgs
from moviad.utilities.get_sizes import params_to_mb, count_params

import torch
from tqdm import tqdm

class StructCoreCL(ContinualModel):

    def __init__(self, structcore_model: StructCore):
        super().__init__(structcore_model)

        # change the structcore memory bank with a set of memory banks, one per task
        self.vad_model.memory_bank = {}
        self.vad_model.train_mu = {}
        self.vad_model.train_sigma = {}
        self.n_samples_per_task = self.vad_model.memory_bank_size

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
            num_workers=0
        )

        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                embedding = self.vad_model(batch)
                # StructCore generates a huge amount of patches at 384x384 (53GB+ of RAM for Varroa).
                # Randomly subsample 10% of the patches to prevent CPU OOM.
                num_patches = embedding.size(0)
                subsample_size = int(num_patches * 0.1)
                indices = torch.randperm(num_patches)[:subsample_size]
                embeddings.append(embedding[indices].cpu())

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings)


            self.vad_model.memory_bank[task_index] = coreset
            
            print(f"Fitting StructCore Calibration for Task {task_index}:")
            descriptors = []
            for batch in tqdm(iter(train_dataloader)):
                if isinstance(batch, tuple):
                    input_tensor = batch[0].to(self.vad_model.device)
                else:
                    input_tensor = batch.to(self.vad_model.device)
                with torch.autocast(device_type="cuda" if "cuda" in str(self.vad_model.device) else "cpu", enabled=True):
                    embedding, batch_size, width, height = self.vad_model.extract_embedding(input_tensor)
                    patch_scores, _ = self.vad_model.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=coreset)
                    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
                    anomaly_maps = self.vad_model.anomaly_map_generator(patch_scores, image_size=input_tensor.shape[2:])
                    for i in range(batch_size):
                        descriptors.append(StructCore.get_structural_descriptor(anomaly_maps[i].squeeze()))
            if len(descriptors) > 0:
                descriptors = torch.stack(descriptors)
                self.vad_model.train_mu[task_index] = descriptors.mean(dim=0)
                self.vad_model.train_sigma[task_index] = descriptors.std(dim=0)
            else:
                self.vad_model.train_mu[task_index] = None
                self.vad_model.train_sigma[task_index] = None

    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass

    def forward(self, batch: torch.Tensor):

      anomaly_maps, pred_scores, mean_distances = [], [], []
      embedding, batch_size, width, height = self.vad_model.extract_embedding(batch)
      image_size = batch.shape[2:]

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

          # Anomaly map
          patch_scores_spatial = patch_scores_2d.reshape(batch_size, 1, width, height)
          batch_anomaly_maps = self.vad_model.anomaly_map_generator(
              patch_scores_spatial, image_size=image_size
          )
          
          # Anomaly score
          pred_score = []
          task_mu = self.vad_model.train_mu.get(task_id, None)
          task_sigma = self.vad_model.train_sigma.get(task_id, None)
          
          for i in range(batch_size):
              descriptor = StructCore.get_structural_descriptor(batch_anomaly_maps[i].squeeze())
              if task_mu is not None and task_sigma is not None:
                  score = StructCore.compute_anomaly_score(descriptor, task_mu, task_sigma)
              else:
                  score = descriptor.sum()
              pred_score.append(score)
          pred_score = torch.stack(pred_score)

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
