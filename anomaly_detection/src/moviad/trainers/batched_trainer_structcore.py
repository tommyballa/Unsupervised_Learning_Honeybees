import torch
from tqdm import tqdm
from typing_extensions import override
from moviad.models.structcore.kmeans_coreset_extractor import MiniBatchKMeansCoresetExtractor
from moviad.models.structcore.structcore import StructCore
from moviad.trainers.trainer_structcore import TrainerStructCore


class BatchStructCoreTrainer(TrainerStructCore):
    def __init__(self, structcore_model: StructCore, train_dataloader: torch.utils.data.DataLoader,
                 test_dataloder: torch.utils.data.DataLoader, device: str, logger=None, cluster_batch_size=16):
        super().__init__(structcore_model, train_dataloader, test_dataloder, device, logger=logger)
        self.cluster_batch_size = cluster_batch_size

    @override
    def train(self):
        self.coreset_extractor = MiniBatchKMeansCoresetExtractor(False, self.device, k=self.structcore_model.k, batch_size=self.cluster_batch_size)
        with torch.no_grad():

            if self.logger is not None:
                self.logger.watch(self.structcore_model)

            print("Embedding Extraction:")
            for batch in tqdm(iter(self.train_dataloader)):
                if isinstance(batch, tuple):
                    embedding = self.structcore_model(batch[0].to(self.device))
                else:
                    embedding = self.structcore_model(batch.to(self.device))

                self.coreset_extractor.partial_fit(embedding)
                del embedding

            coreset = self.coreset_extractor.extract_coreset()

            if self.structcore_model.apply_quantization:
                assert self.structcore_model.product_quantizer is not None, "Product Quantizer not initialized"

                self.structcore_model.product_quantizer.fit(coreset)
                coreset = self.structcore_model.product_quantizer.encode(coreset)


            self.structcore_model.memory_bank = coreset

            print("Fitting StructCore Calibration:")
            descriptors = []
            for batch in tqdm(iter(self.train_dataloader)):
                if isinstance(batch, tuple):
                    input_tensor = batch[0].to(self.device)
                else:
                    input_tensor = batch.to(self.device)
                
                with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=True):
                    embedding, batch_size, width, height = self.structcore_model.extract_embedding(input_tensor)
                    anomaly_maps, _ = self.structcore_model.calculate_anomaly_maps_scores(
                        embedding=embedding,
                        memory_bank=self.structcore_model.memory_bank,
                        batch_size=batch_size,
                        width=width,
                        height=height,
                        image_size=input_tensor.shape[2:]
                    )
                    for i in range(batch_size):
                        descriptors.append(self.structcore_model.get_structural_descriptor(anomaly_maps[i].squeeze()))
            if len(descriptors) > 0:
                descriptors = torch.stack(descriptors)
                self.structcore_model.train_mu = descriptors.mean(dim=0)
                self.structcore_model.train_sigma = descriptors.std(dim=0)


            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.structcore_model)

            if self.logger is not None:
                self.logger.log({
                    "train_loss": 0,
                    "val_loss": 0,
                    "img_roc": img_roc,
                    "pxl_roc": pxl_roc,
                    "f1_img": f1_img,
                    "f1_pxl": f1_pxl,
                    "img_pr": img_pr,
                    "pxl_pr": pxl_pr,
                    "pxl_pro": pxl_pro,
                })

            print("End training performances:")
            print(f"""
                        img_roc: {img_roc} \n
                        pxl_roc: {pxl_roc} \n
                        f1_img: {f1_img} \n
                        f1_pxl: {f1_pxl} \n
                        img_pr: {img_pr} \n
                        pxl_pr: {pxl_pr} \n
                        pxl_pro: {pxl_pro} \n
                    """)
