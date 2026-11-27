"""
StructCore - Anomaly detection on single bees (single bee crops).

Training:      as PatchCore (wide_resnet50_2 features layer1 + layer2, 10%
               of the patches of each image, greedy coreset), plus the
               calibration of the structural descriptor (dispersion, mean
               of the top-1%, total variation of the map) on healthy bees.
               No epoch to choose: memory bank and calibration are
               saved in checkpoints/single_bee/structcore/ (best.pth +
               config.json).
Evaluation:    map = distance from the nearest neighbour in the memory bank,
               image score = diagonal Mahalanobis distance of the
               structural descriptor (computed over the whole map).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO):
    python experiments/plot_heatmaps.py single_bee structcore

Training uses StructCoreCL (a single task) to subsample 10%
of the patches, which is needed to avoid running out of memory.
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import StructCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.structcore_cl import StructCoreCL
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "structcore"
REPO_ROOT    = Path(__file__).resolve().parents[2]
TRAIN_DIR    = DATA_ROOT + "/processed/train_single_bee_batch3+4/cleaned_bees"
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE      = (512, 512)

BACKBONE     = "wide_resnet50_2"
LAYERS       = ["layer1", "layer2"]
MEMORY_BANK_SIZE = 10000
IMG_SIZE     = (384, 384)
BATCH_SIZE   = 8
TEST_BATCH_SIZE = 16
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # half healthy, half with varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Model and data (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def dataset_args():
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    return StructCoreCL(StructCore(feature_extractor=feature_extractor, memory_bank_size=MEMORY_BANK_SIZE))


def model_state(structcore):
    model = structcore.vad_model
    return {
        "memory_bank": model.memory_bank[0].detach().cpu(),
        "train_mu": model.train_mu[0].detach().cpu(),
        "train_sigma": model.train_sigma[0].detach().cpu(),
    }


def load_model_state(structcore, state):
    model = structcore.vad_model
    device = model.device
    model.memory_bank = {0: state["memory_bank"]}
    model.train_mu = {0: state["train_mu"].to(device)}
    model.train_sigma = {0: state["train_sigma"].to(device)}


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_structcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    structcore = build_model(device)
    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)

    # memory bank + descriptor calibration (a single task)
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    structcore.train()
    structcore.start_task(0, train_dataset, training_args)
    structcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    save_checkpoint(CHECKPOINT_DIR, model_state(structcore), script_config(globals()),
                    best_epoch=None, note="memory bank and calibration in a single pass")
    print(f"\nStructCore memory bank and calibration saved to {CHECKPOINT_DIR}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_structcore_single_bee()
