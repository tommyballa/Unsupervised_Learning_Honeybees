"""
PatchCore — Anomaly detection on single bees (Ceschi dataset).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
               val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
               test_heldout: 957 with varroa + 5000 healthy, evaluated once;
               see make_val_split.py).
Training:      frozen ImageNet wide_resnet50_2 features (layer2-3) of healthy
               bees (VarroaDataset, TRAIN), MAX_TRAIN_PATCHES random patches in total
               (same number per image, taken on the bee and not on the black
               background), memory bank reduced to MEMORY_BANK_SIZE patches with a greedy
               coreset. No
               epoch to choose: the memory bank is saved in
               checkpoints/single_bee/patchcore/ (best.pth + config.json).
Evaluation:    map = distance from the nearest neighbour in the memory bank
               (Gaussian blur MAP_SIGMA px), zeroed outside the bee; image
               score = maximum of the map.
Optimisation (sweep: experiments/single_bee/sweep_patchcore.py, batch3+4
dataset, image AUROC at 512 px, bank 30k, 4M candidate patches):
  - previous configuration (layer1-3, blur sigma 2, score = reweighted mean
    of the 5 worst patches): 0.928 (0.932 with 10M candidate patches);
  - wider map blur and score = maximum of the map: sigma 8 -> 0.949,
    beyond sigma 12 it gets worse (at 16-24 pixel AP drops too): the varroa is a
    blob of ~25 px, false positives are isolated peaks of a few patches;
  - layer2-3 instead of layer1-3: 0.956, inference in less than half the time;
  - memory bank 60k: +0.001, not worth the cost;
  - mobilenet_v2 (features.6 + features.13, 20k training images): 0.963
    at image level but lower pixel AP (0.60 vs 0.66 at 128 px).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO), with the same
global normalisation as the other models:
    python experiments/plot_heatmaps.py single_bee patchcore

Training uses PatchCoreCL (a single task) as in the original version.
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import PatchCore
from moviad.models.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.patchcore_cl import PatchCoreCL
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "patchcore"
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATASET_DIR  = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR    = f"{DATASET_DIR}/train/normal"
VAL_DIR      = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR     = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE      = (512, 512)

# frozen ImageNet backbone and feature layers (stride 8 and 16). The default
# is wide_resnet50_2; to compare backbones:
#     PATCHCORE_BACKBONE=mobilenet_v2 python experiments/single_bee/main_patchcore.py
#     PATCHCORE_BACKBONE=mobilenet_v2 python experiments/plot_heatmaps.py single_bee patchcore
# with a non-default backbone, checkpoints and heatmaps go to
# patchcore_<backbone>/, without touching the default ones
BACKBONE_LAYERS = {
    "wide_resnet50_2": ["layer2", "layer3"],
    "resnet18":        ["layer2", "layer3"],
    "mobilenet_v2":    ["features.6", "features.13"],
    "efficientnet_b5": ["features.3", "features.5"],
}
DEFAULT_BACKBONE = "wide_resnet50_2"
BACKBONE     = os.environ.get("PATCHCORE_BACKBONE") or ablation.backbone(DEFAULT_BACKBONE)
LAYERS       = BACKBONE_LAYERS[BACKBONE]
RUN_NAME     = MODEL_NAME if BACKBONE == DEFAULT_BACKBONE else f"{MODEL_NAME}_{BACKBONE}"
MEMORY_BANK_SIZE = 30000
# total candidate patches for the coreset (fp16 on CPU, ~31GB): ~100 per image
MAX_TRAIN_PATCHES = 10_000_000
MAP_SIGMA    = 8            # px at IMG_SIZE, Gaussian blur of the map
# bee mask: non-black pixels (the crop background is black, ~86% of the image),
# closing to fill the dark areas of the body, then a margin around the bee
FG_THRESHOLD = 10 / 255
FG_CLOSING   = 15           # px (kernel side)
FG_DILATION  = 8            # px of margin
IMG_SIZE     = (512, 512)
BATCH_SIZE   = 32
TEST_BATCH_SIZE = 16
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # half healthy, half with varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee_val" / RUN_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / RUN_NAME
# ABL_TRAIN_N / ABL_BACKBONE: see ablation.py
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Model and data (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def dataset_args(test_dir=TEST_DIR):
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        test_path=test_dir,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def _dilate(mask, size):
    return F.max_pool2d(mask, size, stride=1, padding=size // 2)


def bee_foreground(images):
    """(B,1,H,W) bee mask from ImageNet-normalised images."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    rgb = images * std + mean
    mask = (rgb.amax(dim=1, keepdim=True) > FG_THRESHOLD).float()
    mask = 1 - _dilate(1 - _dilate(mask, FG_CLOSING), FG_CLOSING)    # closing
    return _dilate(mask, 2 * FG_DILATION + 1)


class PatchCoreSingleBee(PatchCoreCL):
    """PatchCoreCL with image score = maximum of the map (smoothed and zeroed outside the bee)."""

    def forward(self, batch):
        anomaly_maps, _ = super().forward(batch)
        return anomaly_maps, anomaly_maps.flatten(1).amax(1)


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=MEMORY_BANK_SIZE)
    model.anomaly_map_generator = AnomalyMapGenerator(sigma=MAP_SIGMA)
    return PatchCoreSingleBee(model, max_train_patches=MAX_TRAIN_PATCHES, foreground_fn=bee_foreground)


def model_state(patchcore):
    return {"memory_bank": patchcore.vad_model.memory_bank[0].detach().cpu()}


def load_model_state(patchcore, state):
    patchcore.vad_model.memory_bank = {0: state["memory_bank"]}


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_patchcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    patchcore = build_model(device)
    train_dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))

    # memory bank (a single task)
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    patchcore.train()
    patchcore.start_task(0, train_dataset, training_args)
    patchcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    save_checkpoint(CHECKPOINT_DIR, model_state(patchcore), script_config(globals()),
                    best_epoch=None, note="memory bank built in a single pass")
    print(f"\nPatchCore memory bank saved to {CHECKPOINT_DIR}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_patchcore_single_bee()
