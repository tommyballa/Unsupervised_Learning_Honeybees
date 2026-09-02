from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.single_model import SingleModel
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
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


def train_dinomaly_cl():

    SEEDS = [1]

    for seed in SEEDS:

        set_seed(seed)

        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Dinomaly usa modelli vit, per esempio 'vit_base_patch16_224'
        model = Dinomaly(
            encoder_name="vit_base_patch16_224", 
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9]
        )

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

        continual_model = SingleModel(model)

        # Dinomaly usa total_iters invece di solo epoch per il suo scheduler (vedi dinomaly.py)
        training_args = DinomalyTrainArgs(epochs=5, batch_size=8, total_iters=5000)

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
            ],
            train_args=training_args,
        )

        trainer.train()


if __name__ == "__main__":
    train_dinomaly_cl()
