"""
Dinomaly training optimised for varroa detection on single bees.

Speed optimisations:
  - torch.compile on encoder blocks (frozen) and decoder blocks
  - AMP bfloat16 on forward + loss
  - batch_size=32 (an A6000 48GB handles it comfortably)
  - Evaluation every EVAL_EVERY iterations
  - persistent_workers + pin_memory in the DataLoaders
  - cudnn.benchmark=True (fixed input size)
  - zero_grad(set_to_none=True) + non_blocking transfers

Varroa quality optimisations:
  - Geometric data augmentation (flips, rotation)
  - GT masks, metrics and heatmaps at 512x512 (the crop resolution)

Alignment with the official Dinomaly implementation:
  - GLOBAL cosine loss with hard mining (components.global_cosine_hm_percent)
  - map smoothed with a Gaussian (kernel 5, sigma 4) and image score = mean
    of the top-1% pixels (Dinomaly(max_ratio=0.01))
  - evaluation in fp32 (training stays in bf16)

Dataset: single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
test_heldout: 957 with varroa + 5000 healthy, evaluated once;
see make_val_split.py).
With 102k images an epoch is ~3200 iterations: training is in iterations
(TOTAL_ITERS = 10k as in the official repo, cosine scheduler over TOTAL_ITERS).
On train_single_bee_merged (23k images) the official ViT-B configuration
(layers 2-9 in two groups, ~7k iterations) had given img AUROC 0.960 vs
0.949 for layers 2-5 (pxl AP 0.574 vs 0.587).

Best: every EVAL_EVERY iterations (and at the last one) metrics on the validation set; the
best configuration (img_roc_auc, ties broken by pxl_avg_prec) is saved in
checkpoints/single_bee/dinomaly/ (best.pth = bottleneck + decoder, the DINOv2
encoder is frozen; config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee dinomaly
"""

from moviad.datasets.varroa.paths import DATA_ROOT
from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import os
import random
from pathlib import Path
import numpy as np

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME     = "dinomaly"
# vit_small_patch14_reg4_dinov2 (same layers, as in the official repo): ablation (ABL_BACKBONE)
ENCODER_NAME   = ablation.backbone("vit_base_patch14_reg4_dinov2")
# official Dinomaly configuration for ViT-B: 8 intermediate layers fused in two groups
TARGET_LAYERS  = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
MAX_RATIO      = 0.01   # image score = mean of the top-1% pixels (as in the official repo)
IMG_SIZE       = (518, 518)
TOTAL_ITERS    = 10_000  # training iterations (~3 epochs)
BATCH_SIZE     = 32
EVAL_EVERY     = 1000   # evaluate every N iterations (and always at the last one)
SEED           = 1
NUM_WORKERS    = 8

REPO_ROOT      = Path(__file__).resolve().parents[2]
DATASET_DIR    = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR      = f"{DATASET_DIR}/train/normal"
VAL_DIR        = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR       = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE        = (512, 512)

NUM_HEATMAPS   = 20      # half healthy, half with varroa
OUTPUT_DIR     = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = Path(os.environ.get("DINOMALY_CKPT_DIR",
                                    REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME))
if ablation.ACTIVE:
    CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is safe with a fixed input size and speeds up convolutions
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Dataset wrapper with data augmentation
# ──────────────────────────────────────────────────────────────────────
class AugmentedTrainDataset(Dataset):
    """
    Wraps the VarroaDataset (which caches the transformed tensors) and applies
    random geometric augmentation on every access.

    The augmentation works on already-normalised tensors: geometric
    transformations (flip, rotation) do not alter the value distribution.
    This is essential for varroa detection because bees can
    appear in any orientation.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15, fill=0),
            ], p=0.3),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image = self.base[idx]  # cached tensor (C, H, W)
        return self.aug(image)


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


def build_model(device):
    model = Dinomaly(
        encoder_name=ENCODER_NAME,
        target_layers=TARGET_LAYERS,
        fuse_layer_encoder=FUSE_LAYER_ENCODER,
        fuse_layer_decoder=FUSE_LAYER_DECODER,
        # the decoder uses LinearAttention2, which ignores the attention mask:
        # neighbour masking would have no effect (and is not used in the official repo)
        mask_neighbor_size=0,
        img_size=IMG_SIZE,
        max_ratio=MAX_RATIO,
    )
    return model.to(device)


def model_state(model):
    # frozen DINOv2 encoder: bottleneck + decoder are enough (without the torch.compile prefixes)
    return clean_state_dict(model.trainable.state_dict())


def load_model_state(model, state):
    model.trainable.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Direct training (bypasses ContinualTrainer for maximum speed)
# ──────────────────────────────────────────────────────────────────────
def train_dinomaly():

    set_seed(SEED)
    torch.set_float32_matmul_precision('high')

    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(device)

    # ── torch.compile on the compute blocks ──────────────────────────
    # The encoder is frozen with a fixed input size → ideal for compilation
    print("Compiling encoder blocks...")
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            model.encoder.blocks[i] = torch.compile(blk)

    # Decoder and bottleneck: compiled to speed up the backward pass
    print("Compiling decoder and bottleneck blocks...")
    for i, blk in enumerate(model.decoder):
        model.decoder[i] = torch.compile(blk)
    for i, blk in enumerate(model.bottleneck):
        model.bottleneck[i] = torch.compile(blk)

    # ── Dataset ──────────────────────────────────────────────────────
    # Training dataset with geometric augmentation
    train_dataset = AugmentedTrainDataset(ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN)))

    # Test dataset (no augmentation)
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Training args ────────────────────────────────────────────────
    training_args = DinomalyTrainArgs(
        epochs=-(-TOTAL_ITERS // len(train_loader)),
        batch_size=BATCH_SIZE,
        total_iters=TOTAL_ITERS,
    )
    training_args.init_train(model)
    # the "best_epoch" saved by the tracker is the number of iterations
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    # ── Training loop (iteration-based) ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Dinomaly | {TOTAL_ITERS} iterations | {len(train_loader)} batches/epoch")
    print(f"Batch size: {BATCH_SIZE} | Eval every {EVAL_EVERY} iterations")
    print(f"{'='*60}\n")

    def batches():
        while True:
            yield from train_loader

    model.train()
    running_loss = 0.0
    for it, batch in enumerate(tqdm(batches(), total=TOTAL_ITERS), start=1):
        running_loss += model.train_step(batch, training_args)

        # Evaluation (every EVAL_EVERY iterations and at the last one) + saving the best checkpoint
        if it % EVAL_EVERY == 0 or it == TOTAL_ITERS:
            n = EVAL_EVERY if it % EVAL_EVERY == 0 else it % EVAL_EVERY
            print(f"\nITER: {it} | Loss: {running_loss / n:.6f}")
            running_loss = 0.0
            maps, scores, labels, masks = run_inference(model, eval_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(it, results, lambda: model_state(model))
            del maps, scores, labels, masks
            model.train()

        if it == TOTAL_ITERS:
            break

    print(f"\nBest Dinomaly — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_dinomaly()
