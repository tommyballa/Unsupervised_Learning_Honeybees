from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from torch.utils.data import Subset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
import torch
from torchvision import transforms
import random
import numpy as np

import argparse


BACKBONES = [
    "mcunet-in3",
    "micronet-m0",
    "micronet-m1",
    "micronet-m2",
    "micronet-m3",
    "phinet_1.2_0.5_6_downsampling",
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
    "mobilenet_v2",
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="E.g. FT_stfpm, MT_stfpm, replay_stfpm", choices=["FT_stfpm", "MT_stfpm", "replay_stfpm"])
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="E.g. wide_resnet50_2", choices=BACKBONES)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", type=int, required=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    return args



def train_stfpm_FT(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=20, batch_size=16, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    model = STFPM(teacher, student)
    model.to(device)
    continual_model = FineTuning(model)

    trainer = ContinualTrainer(
        dataset,
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
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size),
        logger=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def train_stfpm_multi_task(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=50, batch_size=16, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)

    train_dataset, test_dataset = dataset.get_all_tasks_data()

    model = STFPM(teacher, student)
    model.to(device)

    training_args = STFPMTrainArgs(epochs=epochs, batch_size=batch_size)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=None,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def train_stfpm_replay(dataset, backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], epochs=10, batch_size=8, device="cpu"):

    teacher = CustomFeatureExtractor(backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(backbone, layers, device, frozen=False)
    model = STFPM(teacher, student).to(device)

    continual_model = Replay(model, 100, 0.5)

    trainer = ContinualTrainer(
        dataset,
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
        training_args=STFPMTrainArgs(epochs=epochs, batch_size=batch_size),
        logger=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def main():
    args = get_test()
    numEpochs = args.epochs
    batch_size = args.batch_size
    backbone = args.backbone
    model = args.model

    set_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device is not None:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else device

    dataset_args = {
        "dataset_path": "./data",
        "img_size": (224, 224),
        "gt_mask_size": (224, 224),
        "image_transform_list": None,
    }

    # Continual learning with video folders as incremental tasks
    continual_dataset = ContinualDataset(
        dataset_arguments=DatasetArguments(**dataset_args),
        dataset_class=VarroaDataset,
        # Uses the training video folders as CL tasks
        # For a single initial task you can pass categories=["varroa"]
        # For incremental CL, leave None to use all video folders
    )

    if backbone == "mcunet-in3":
        layers = ["2", "6", "14"]
    else:
        layers = ["layer1", "layer2", "layer3"]

    if model == "FT_stfpm":
        train_stfpm_FT(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model == "MT_stfpm":
        train_stfpm_multi_task(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    elif model == "replay_stfpm":
        train_stfpm_replay(dataset=continual_dataset, backbone=backbone, layers=layers, epochs=numEpochs, batch_size=batch_size, device=device)
    else:
        raise NotImplementedError(f"Model {model} not implemented in tests.")



if __name__ == "__main__":
    main()