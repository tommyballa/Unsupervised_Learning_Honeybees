"""
RD4AD (Reverse Distillation) - Anomaly detection on single bees (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
               val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
               test_heldout: 957 with varroa + 5000 healthy, evaluated once;
               see make_val_split.py).
               Re-optimised on this dataset (sweep_rd4ad.py, runs sam3_*,
               sweep metrics: pixel AP at 128 px). The merged-dataset recipe
               (724 steps, layers 1+2, sigma 4) stops at img AUROC 0.914 /
               pixel AP 0.27 (peak at 273 steps, then ~0.88):
                 - 400 steps with cosine, evaluation every 40: the peak is between 160 and
                   400 steps but fluctuates by ±0.02 between neighbouring evaluations;
                 - map from layer 2 only with sigma 8 px (= 4 at 256): ~0.94-0.95 /
                   0.33-0.37 (layers 1+2: ~0.94 / 0.35-0.38; layers 0+1+2: 0.86);
                   sigma 12, 16 or 24 px and top-k score do not exceed ~0.945;
                 - lr 0.001: early peak (0.943 at 160 steps) and then a drop to 0.92.
               With this script (official metrics at 512): step 400 img_roc_auc
               0.953 / pxl_avg_prec 0.362, AU-PRO 0.989 (0.93-0.95 from 160 steps).
               Below, the history of the sweep on the merged dataset (23k bees).
Training:      normal images of single bees (VarroaDataset, TRAIN split),
               with geometric augmentation (flip) because bees can have
               any orientation.
Optimisation (sweep: experiments/single_bee/sweep_rd4ad.py):
  - native resolution 512 (varroa mites are small): img AUROC ~0.90 at 256,
    ~0.96 at 384, ~0.978 at 512;
  - map from layers 1 and 2 only (layer 0 is noisy: img AP 0.85 vs ~0.75);
  - short training: with 23k images RD4AD over-generalises already after ~1 epoch
    (the decoder learns to reconstruct the anomalies too and the metrics drop),
    hence few epochs with cosine LR and frequent evaluation;
  - weight EMA (EMA_DECAY) available but disabled: it stabilises long
    runs, but lags behind at the early peak (384: img AUROC 0.954 vs 0.962);
  - training in bf16 (autocast), evaluation in fp32.
Best:          every EVAL_INTERVAL steps (and at the end) metrics on the validation set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/rd4ad/ (best.pth +
               config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee rd4ad
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.rd4ad.loss_functions import rd4ad_cosine_loss
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "rd4ad"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE       = (512, 512)

BACKBONE      = ablation.backbone("wide_resnet50_2")   # resnet18: ablation (ABL_BACKBONE)
IMG_SIZE      = (512, 512)   # RD4AD requires multiples of 32
TRAIN_STEPS   = 400          # total steps: peak between 160 and 400 steps (merged: 724 = 1/2 epoch)
BATCH_SIZE    = 16
LR            = 0.005        # paper: Adam lr 0.005, betas (0.5, 0.999)
BETAS         = (0.5, 0.999)
WARMUP_FRAC   = 0.02         # linear warmup, then cosine decay to 0
EMA_DECAY     = 0.0          # EMA of the bn + decoder weights (0 = off: at the early peak the EMA lags behind)
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 40           # evaluate every N steps and always at the end
BLUR_SIGMA    = 8.0          # blur sigma on the map at IMG_SIZE (= 4 at 256; merged: 4)
LAYERS        = (2,)         # layers summed in the map (None = all 3; merged: (1, 2))
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # half healthy, half with varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class AugmentedTrainDataset(Dataset):
    """Applies random flips to the (already normalised) images of the training dataset."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.aug(self.base[idx])


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


def build_model(device):
    model = RD4AD(backbone_name=BACKBONE, input_size=IMG_SIZE,
                  blur_sigma=BLUR_SIGMA, layers=LAYERS).to(device)
    # the encoder (teacher) is frozen: the optimizer updates only bottleneck and decoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


def model_state(model):
    # the encoder is the frozen ImageNet wide_resnet50_2: bottleneck and decoder are enough
    return {"bn": clean_state_dict(model.bn.state_dict()),
            "decoder": clean_state_dict(model.decoder.state_dict())}


def load_model_state(model, state):
    model.bn.load_state_dict(state["bn"])
    model.decoder.load_state_dict(state["decoder"])


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def update_ema(ema_model, model, step):
    # reduced decay in the first steps (like timm ModelEmaV2 with warmup)
    decay = min(EMA_DECAY, (1 + step) / (10 + step))
    for ema_mod, mod in ((ema_model.bn, model.bn), (ema_model.decoder, model.decoder)):
        for e, m in zip(ema_mod.state_dict().values(), mod.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(m.detach(), alpha=1 - decay)
            else:
                e.copy_(m)


def train_rd4ad_single_bee():
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
    ema_model = copy.deepcopy(model) if EMA_DECAY > 0 else model
    ema_model.encoder = model.encoder  # shared frozen encoder

    train_dataset = AugmentedTrainDataset(ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN)))
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # the encoder (teacher) is frozen: the optimizer updates only bottleneck and decoder
    optimizer = torch.optim.Adam(list(model.decoder.parameters()) + list(model.bn.parameters()),
                                 lr=LR, betas=BETAS)
    total_steps = TRAIN_STEPS
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training RD4AD single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epochs "
          f"| {len(train_loader)} batches/epoch | evaluation every {EVAL_INTERVAL} steps")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            model.train()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                teacher = model.encoder(batch)
                student = model.decoder(model.bn(teacher))
            loss = sum(rd4ad_cosine_loss(t.float(), s.float()) for t, s in zip(teacher, student))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if ema_model is not model:
                update_ema(ema_model, model, step)
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoch {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(ema_model, val_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(ema_model))
            if step >= total_steps:
                break

    print(f"\nBest RD4AD (\"epoch\" = training step) — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_rd4ad_single_bee()
