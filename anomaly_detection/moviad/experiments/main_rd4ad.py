from moviad.models import RD4AD
from moviad.models.rd4ad.rd4ad import RD4ADTrainArgs
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
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


def train_rd4ad_cl():

    SEEDS = [1, 3, 1024]

    for seed in SEEDS:

        set_seed(seed)

        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        model = RD4AD(backbone_name="wide_resnet50_2").to(device)

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

        continual_model = Replay(model, memory_size=100, replay_ratio=0.5)

        training_args = RD4ADTrainArgs(epochs=30, batch_size=32, evaluation_epoch_interval=40)
        training_args.init_train(model)

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


if __name__ == "__main__":
    train_rd4ad_cl()
