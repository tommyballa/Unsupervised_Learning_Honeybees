from moviad.datasets.vad_dataset import VADDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.structcore.structcore import StructCore
from moviad.models.training_args import TrainingArgs

import torch
from tqdm import tqdm

class StructCoreFineTuningPP(ContinualModel):

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass    

    def train_task(self, task_index: int, train_dataset, eval_dataset, metrics, device, logger = None, train_args:TrainingArgs = None):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4
        )

        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                embedding = self.vad_model(batch)
                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            if task_index > 0:
                embeddings = torch.cat([embeddings, self.vad_model.memory_bank], dim=0)

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings)


            self.vad_model.memory_bank = coreset
            
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
                self.vad_model.train_mu = descriptors.mean(dim=0)
                self.vad_model.train_sigma = descriptors.std(dim=0)

    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass