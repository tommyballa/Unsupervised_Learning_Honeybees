from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.datasets.varroa.varroa_domain_shift_dataset import VarroaDomainShiftDataset
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

def train_stfpm_cl():
    SEEDS = [1, 3, 1024]
    
    for seed in SEEDS:
        set_seed(seed)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # wide_resnet50_2 is heavy, resnet18 is often enough for stfpm and trains faster
        teacher = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device, frozen=True)
        student = CustomFeatureExtractor("resnet18", ["layer1", "layer2", "layer3"], device, frozen=False)
        model = STFPM(teacher, student).to(device)

        from torchvision import transforms
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        
        dataset_args = DatasetArguments(
            dataset_path="./data",
            img_size=(256, 256),  # reduced for faster training and memory stability
            gt_mask_size=(256, 256),
            image_transform_list=[
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                # removed gaussian blur to preserve small mite details
                imagenet_normalize,
            ],
        )

        continual_dataset = ContinualDataset(
            dataset_args,
            VarroaDomainShiftDataset,
            categories=["varroa_clean", "varroa_dirty_lens", "varroa_low_light"],
        )

        # increased memory size to retain more knowledge across shifts
        continual_model = Replay(model, memory_size=200, replay_ratio=0.5)

        # increased epochs for better student convergence
        training_args = STFPMTrainArgs(epochs=100, batch_size=8)
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
            ],
            training_args=training_args,
            logger=None,
        )

        trainer.train()

if __name__ == "__main__":
    train_stfpm_cl()