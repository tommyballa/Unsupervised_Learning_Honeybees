"""
FastFlow - Anomaly detection on single bees (single bee crops).

Training:      frozen ImageNet wide_resnet50_2 features (layer1-3) on the crops
               at native 512x512 resolution, trainable LayerNorms per scale and a
               2D normalizing flow per scale, maximising the likelihood of healthy
               bees (VarroaDataset, TRAIN). Adam lr 3e-4, wd 1e-5, one-cycle cosine.
               Differences from standard FastFlow (see sweep_fastflow.py):
               - the crops have a black background: the loss is the per-pixel NLL
                 averaged only over bee pixels (otherwise the almost constant
                 background dominates the loss and the map lights up on background and edges);
               - few steps (STEPS): the flow peaks after 50-150 steps and then
                 gets worse (it also learns the varroa as "likely");
               - map = mean over scales of mean(z^2), restricted to the bee eroded by
                 FG_ERODE px, smoothed (sigma BLUR_SIGMA); image score = maximum
                 of the map.
Optimisation (sweep_fastflow.py, logs/fastflow_sweep/):
               - merged dataset (23k bees, varroa ~25 px): 512 px and masked loss
                 gave the largest gains; NLL map layer2+3, sigma 8, erosion 8
                 -> img AUROC 0.962.
               - final SAM3 dataset (runs f1-f4): 0.913 with the merged-dataset map.
                 The varroa are larger (~43 px) and the edges of the SAM3 silhouettes
                 noisier: z^2 on all scales, sigma 16, erosion 24 px -> 0.983
                 (a 24 px erosion keeps 98% of the varroa pixels inside the bee).
                 lr 1e-4 (0.981), loss on the bee dilated by 16 px (0.979) and a reduced flow
                 4 steps / hidden 0.5 (0.971) do not improve.
Best:          every EVAL_INTERVAL steps (and at the last one) metrics on the validation set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/fastflow/ (best.pth +
               config.json). best.pth is the full state dict of the model, the
               same format read by diagnose_fastflow_varroa.py.
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee fastflow
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "fastflow"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# False = subset of 5,000 healthy + all 1,279 with varroa (like the other
# models); True = the whole test set (24,297 healthy): AP and F1 depend on the anomaly share
FULL_TEST     = False
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE       = (512, 512)

# cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2
BACKBONE      = ablation.backbone("wide_resnet50_2")   # resnet18: ablation (ABL_BACKBONE)
IMG_SIZE      = (512, 512)   # native resolution: at 256 the varroa covers ~14x14 px
STEPS         = 200          # training steps (1 epoch = ~12800 steps)
BATCH_SIZE    = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-5
EVAL_INTERVAL = 25           # evaluate every N steps (and always at the last one)
NUM_WORKERS   = 8

# foreground: bee crops on a black background
FG_THRESHOLD  = 0.1          # max RGB intensity (0-1) above which a pixel is bee
MASKED_LOSS_DILATE = 4       # loss on the bee pixels dilated by N px
FG_ERODE      = 24           # map restricted to the bee eroded by N px (unstable silhouette edge)
# anomaly map
MAP_TYPE      = "z2"         # "z2" (original FastFlow) | "nll" (0.5 z^2 - log-det per pixel)
MAP_SCALES    = None         # None = all (layer1-3)
BLUR_SIGMA    = 16.0
SEED          = 1

NUM_HEATMAPS  = 20           # half healthy, half with varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is safe with a fixed input size and speeds up convolutions
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Model and data (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def dataset_args(test_dir=TEST_DIR):
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        test_path=test_dir,
        full_test=FULL_TEST,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_model(device):
    # LayerNorm per scale + normalizing flow, as in the paper
    return create_fastflow(
        img_shape=IMG_SIZE, backbone_name=BACKBONE, device=device, normalize=True,
        blur_sigma=BLUR_SIGMA, map_type=MAP_TYPE, map_scales=MAP_SCALES,
        foreground_threshold=FG_THRESHOLD, foreground_erode=FG_ERODE,
        masked_loss_dilate=MASKED_LOSS_DILATE, backbone_amp=True,
        image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
    )


def model_state(model):
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_fastflow_single_bee():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(device)

    train_dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Optimizer (flow + LayerNorm), per-step scheduler and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=STEPS, pct_start=0.05,
        anneal_strategy="cos", div_factor=10, final_div_factor=100)
    training_args = FastFlowTrainArgs(epochs=1, batch_size=BATCH_SIZE,
                                      optimizer=optimizer, scheduler=scheduler)
    training_args.init_train(model)
    # the best checkpoint is indexed by step (config.json: "best_epoch" = step)
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training FastFlow single-bee | {STEPS} steps | {len(train_loader)} batches/epoch")
    print(f"{'='*60}\n")

    def batches():
        while True:
            yield from train_loader

    model.train()
    running = 0.0
    for step, batch in enumerate(batches(), start=1):
        running += model.train_step(batch, training_args)
        scheduler.step()

        if step % EVAL_INTERVAL == 0 or step == STEPS:
            print(f"Step {step}/{STEPS} — loss: {running / EVAL_INTERVAL:.4f}, "
                  f"lr: {scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
            maps, scores, labels, masks = run_inference(model, val_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            del maps
            print_metrics(results)
            tracker.update(step, results, lambda: model_state(model))
            model.train()
        if step >= STEPS:
            break

    print(f"\nBest FastFlow — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_fastflow_single_bee()
