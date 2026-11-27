"""
CFA (Coupled-hypersphere-based Feature Adaptation) - Anomaly detection on
single bees (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
               val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
               test_heldout: 957 with varroa + 5000 healthy, evaluated once;
               see make_val_split.py).
               The sweep below was done on the batch3+4 dataset (23k bees):
               the configuration is the same, with BANK_PATCHES_PER_IMAGE reduced
               to 16 (same total number of patches for KMeans, ~1.6M) and
               training defined in steps (TRAIN_STEPS) instead of epochs.
Training:      frozen ImageNet wide_resnet50_2 backbone, memory bank
               built from the features (initial descriptor) of healthy bees,
               patch descriptor trained with the soft-boundary loss (AdamW,
               wd 5e-4, amsgrad). Image score = maximum of the smoothed
               heatmap (paper).
Optimisation (sweep: experiments/single_bee/sweep_cfa.py; image AUROC,
the "without training" numbers use the initial descriptor):
  - the paper's CFA (memory bank = per-position mean of the features, 384 px,
    layers 1-2, lr 1e-3) stops at ~0.72: the bees are not aligned and 86%
    of the crop is black background, so the bank is almost all background;
  - memory bank = KMeans centroids of patches sampled only on the bee
    (BANK_INIT "kmeans" + bee_foreground), loss and map only on the bee:
    0.80 without training with 16k centroids (0.73 with 4k); 0.85 at 512 px;
  - K = 1 (distance from the nearest centroid only): 0.87 vs 0.85 with
    K = 5, the softmin over the K neighbours pollutes the score;
  - layers 2-3 instead of 1-2: 0.94 with 32k centroids (layers 1-3: 0.93;
    64k centroids: +0.001, not worth twice the cost);
  - crops upscaled to 768 px (varroa mites are small and layers 2-3 have
    stride 8): 0.966 (0.965 at 1024 px, with worse maps);
  - short training with a low lr: with 23k images the paper's lr (1e-3, meant
    for ~500 steps on ~250 images) collapses the descriptor, which squashes
    the features onto the bank (from 0.80 to 0.62 in 360 steps; with 1e-4 after ~90 steps).
    With lr 1e-5 the adaptation goes from 0.966 to 0.973 with a peak between 260 and
    430 steps, then drops: a few hundred steps with cosine LR and frequent
    evaluation (step 0 = no adaptation).
Best:          every EVAL_INTERVAL steps (and at the end) metrics on the validation set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/cfa/ (best.pth +
               config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee cfa
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.cfa.cfa import CFA, CFATrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "cfa"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE       = (512, 512)

# layers at stride 8 and 16 and concatenated channels; other backbones only for
# the ablation (ABL_BACKBONE, see ablation.py)
BACKBONE_LAYERS = {
    "wide_resnet50_2": (["layer2", "layer3"], 1536),       # 512 + 1024
    "resnet18":        (["layer2", "layer3"], 384),        # 128 + 256
    "mobilenet_v2":    (["features.6", "features.13"], 128),  # 32 + 96
}
BACKBONE      = ablation.backbone("wide_resnet50_2")
LAYERS, FEATURE_CHANNELS = BACKBONE_LAYERS[BACKBONE]
# gamma_d=2: patch descriptor channels halved (1536 -> 768)
GAMMA_D       = 2
IMG_SIZE      = (768, 768)   # 512x512 crops upscaled: 96x96 map (layer2, stride 8)
# memory bank: KMeans centroids of BANK_PATCHES_PER_IMAGE random patches per
# image taken on the bee ("mean" = per-position mean, as in the paper)
BANK_INIT     = "kmeans"
BANK_SIZE     = 32768
BANK_PATCHES_PER_IMAGE = 16   # 102k images -> ~1.6M patches (64 with the 23k of batch3+4)
if ablation.TRAIN_N:
    # fewer bees: same ~1.6M candidate patches for KMeans, at most 256 per bee
    BANK_PATCHES_PER_IMAGE = min(256, -(-BANK_PATCHES_PER_IMAGE * 102_303 // ablation.TRAIN_N))
K             = 1            # nearest centroids used by the loss (attraction) and the score
J             = 3            # next centroids used by the loss (repulsion)
# bee mask: non-black pixels (the crop background is black, ~86% of the image),
# closing to fill the dark areas of the body, then a margin around the bee
# (in pixels at GT_SIZE, as in PatchCore)
FG_THRESHOLD  = 10 / 255
FG_CLOSING    = 15
FG_DILATION   = 8
TRAIN_STEPS   = 522          # the peak is after a few hundred steps (= 0.36 epochs of batch3+4)
BATCH_SIZE    = 16
TEST_BATCH_SIZE = 32
LR            = 1e-5         # paper: 1e-3 (here it collapses the descriptor)
WEIGHT_DECAY  = 5e-4
WARMUP_FRAC   = 0.02         # linear warmup, then cosine decay to 0
EVAL_INTERVAL = 87           # evaluate every N steps and always at the end
BLUR_SIGMA    = 4.0          # blur sigma on the map at IMG_SIZE
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # half healthy, half with varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


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
def dataset_args(test_dir=TEST_DIR):
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        test_path=test_dir,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
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
    scale = images.shape[-1] / GT_SIZE[1]
    closing = 2 * round(FG_CLOSING * scale / 2) + 1
    mask = 1 - _dilate(1 - _dilate(mask, closing), closing)    # closing
    return _dilate(mask, 2 * round(FG_DILATION * scale) + 1)


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = CFA(
        feature_extractor=feature_extractor,
        backbone=BACKBONE,
        gamma_d=GAMMA_D,
        feature_maps_channels=FEATURE_CHANNELS,
        K=K,
        J=J,
        bank_init=BANK_INIT,
        bank_patches_per_image=BANK_PATCHES_PER_IMAGE,
        bank_size=BANK_SIZE,
        blur_sigma=BLUR_SIGMA,
        foreground_fn=bee_foreground,
    )
    model.to(device)
    return model


def model_state(model):
    # patch descriptor, radius r and memory bank (the frozen backbone is not in the state dict)
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    # in a new model the memory bank does not exist yet (it is created before training)
    model.memory_bank = torch.nn.Parameter(state["memory_bank"].clone().to(model.device),
                                           requires_grad=False)
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_cfa_single_bee():
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
        val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # AdamW(wd 5e-4, amsgrad) on patch descriptor and radius + soft-boundary loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    training_args = CFATrainArgs(epochs=math.ceil(TRAIN_STEPS / len(train_loader)), batch_size=BATCH_SIZE, optimizer=optimizer)
    training_args.init_train(model)
    total_steps = TRAIN_STEPS
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\nMemory bank ({BANK_INIT}): features of healthy bees with the initial descriptor")
    model.train()
    model.memory_bank = torch.nn.Parameter(model.initialize_memory_bank(train_loader), requires_grad=False)

    print(f"\n{'='*60}")
    print(f"Training CFA single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epochs "
          f"| {len(train_loader)} batches/epoch | evaluation every {EVAL_INTERVAL} steps")
    print(f"{'='*60}\n")

    def evaluate(step):
        maps, scores, labels, masks = run_inference(model, val_loader, device)
        results = compute_metrics(maps, scores, labels, masks, with_pro=False)
        del maps, masks
        print_metrics(results)
        tracker.update(step, results, lambda: model_state(model))

    # step 0: memory bank with the initial descriptor (no adaptation)
    print("Step 0 (initial descriptor)")
    evaluate(0)

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            running_loss += model.train_step(batch, training_args)
            scheduler.step()
            step += 1; n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoch {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                evaluate(step)
            if step >= total_steps:
                break

    print(f"\nBest CFA (\"epoch\" = training step) — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_cfa_single_bee()
