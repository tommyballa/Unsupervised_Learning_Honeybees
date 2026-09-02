from moviad.models import Padim
from moviad.models.padim.padim import PadimTrainArgs
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.padim_cl import PadimCLUnimodal, PadimCLMultimodal
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


def train_padim_cl():

    SEEDS = [1]

    for seed in SEEDS:

        set_seed(seed)

        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        model = Padim(
            backbone_model_name="mobilenet_v2",
            layers_idxs=(4, 7, 10),
            device=device,
        ).to(device)

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

        continual_model = PadimCLUnimodal(model)

        training_args = PadimTrainArgs(epochs=1, batch_size=32)

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
            training_args=training_args,
            logger=None,
        )

        trainer.train()

        # Salvataggio del checkpoint su disco
        import os
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint_path = "./checkpoints/padim_varroa.pt"
        torch.save(
            {
                "gauss_mean": model.gauss_mean,
                "gauss_cov": model.gauss_cov,
                "random_dimensions": model.random_dimensions.cpu(),
                "backbone_model_name": model.backbone_model_name,
                "layers_idxs": (4, 7, 10),
            },
            checkpoint_path,
        )
        print(f"\n💾 Checkpoint salvato con successo in: {checkpoint_path}")


if __name__ == "__main__":
    train_padim_cl()
