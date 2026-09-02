from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.single_model import SingleModel
from moviad.datasets.varroa.varroa_domain_shift_dataset import VarroaDomainShiftDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
import torch
import random
import numpy as np
import os


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

        # Dinomaly usa modelli vit, per esempio 'vit_base_patch14_reg4_dinov2'
        model = Dinomaly(
            encoder_name="vit_base_patch14_reg4_dinov2", 
            target_layers=[2, 3, 4, 5],
            fuse_layer_encoder=[[0, 1, 2, 3]],
            img_size=(336, 336)
        )
        model = model.to(device)

        args = {
            "dataset_path": "./data",
            "img_size": (336, 336),
            "gt_mask_size": (336, 336),
            "image_transform_list": None,
        }

        # Solo la prima task (varroa_clean) con domain shift
        continual_dataset = ContinualDataset(
            DatasetArguments(**args),
            VarroaDomainShiftDataset,
            categories=["varroa_clean"],
        )

        # Crea la cartella per i checkpoint
        os.makedirs("./checkpoints", exist_ok=True)
        
        continual_model = SingleModel(model, save_directory_path="./checkpoints")

        # Dinomaly usa total_iters invece di solo epoch per il suo scheduler (vedi dinomaly.py)
        # Il batch size è stato ridotto a 8 per evitare il CUDA Out Of Memory
        training_args = DinomalyTrainArgs(epochs=100, batch_size=6, total_iters=23600)

        trainer = ContinualTrainer(
            continual_dataset,
            continual_model,
            device,
            metrics=[
                RocAuc(MetricLvl.IMAGE),
                AvgPrec(MetricLvl.IMAGE),
                F1(MetricLvl.IMAGE),
            ],
            training_args=training_args,
        )

        trainer.train()


if __name__ == "__main__":
    train_dinomaly_cl()
