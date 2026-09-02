from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models import StructCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.structcore_cl import StructCoreCL
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_structcore_cl():

    SEEDS = [1]

    for seed in SEEDS:

        set_seed(seed)

        # Usa l'accelerazione GPU Apple (MPS) se disponibile, altrimenti CPU
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Usiamo mobilenet_v2 e un solo layer intermedio leggero per abbattere i 1792 canali
        feature_extractor = CustomFeatureExtractor("mobilenet_v2", ["features.7"], device, frozen=True)
        model = StructCore(feature_extractor=feature_extractor)

        args = {
            "dataset_path": "./data",
            "img_size": (224, 224),
            "gt_mask_size": (224, 224),
            "image_transform_list": None,
        }

        # Singola task: tutto il dataset aggregato
        continual_dataset = ContinualDataset(
            DatasetArguments(**args),
            VarroaDataset,
            categories=["varroa"],
        )

        continual_model = StructCoreCL(model)

        trainer = ContinualTrainer(
            continual_dataset,
            continual_model,
            device,
            metrics=[
                RocAuc(MetricLvl.IMAGE),
                RocAuc(MetricLvl.PIXEL),
                AvgPrec(MetricLvl.IMAGE),
                AvgPrec(MetricLvl.PIXEL),
                F1(MetricLvl.IMAGE),
                F1(MetricLvl.PIXEL),
                ProAuc(MetricLvl.PIXEL),
            ],
            training_args=TrainingArgs(
                batch_size=32,
                epochs=1,
            ),
            logger=None,
        )

        trainer.train()

        # Salvataggio del checkpoint della memory bank di StructCore
        import os
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint_path = "./checkpoints/structcore_varroa.pt"
        torch.save(
            {
                "memory_bank": model.memory_bank.cpu() if model.memory_bank is not None else None,
            },
            checkpoint_path,
        )
        print(f"\n💾 Checkpoint salvato con successo in: {checkpoint_path}")


if __name__ == "__main__":
    train_structcore_cl()
