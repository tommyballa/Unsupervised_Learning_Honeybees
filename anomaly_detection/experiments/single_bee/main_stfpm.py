"""
STFPM (Student-Teacher Feature Pyramid Matching) - Anomaly detection on
single bees (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
               val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
               test_heldout: 957 with varroa + 5000 healthy, evaluated once;
               see make_val_split.py).
               Re-optimised on this dataset (sweep_stfpm.py, runs sam3_*,
               sweep metrics: pixel AP at 128 px). The merged-dataset recipe
               (layer2+layer3, cosine over 1448 steps) stops at img AUROC
               0.967 / pixel AP 0.41 with a peak at 543 steps and then drops. Here the
               student generalises even earlier and layer2 becomes noisy:
                 - map from layer3 only (MAP_LAYERS = [2]), blur sigma 8 px:
                   0.981-0.983 / 0.55-0.57 between 300 and 420 steps (layer2+layer3:
                   0.974 / 0.48), score = maximum (top-k: +0.001);
                 - 600 steps with cosine (0.978-0.983 from 300 steps to the end);
                 - lr 0.1 over 1200 steps: worse and unstable (peak 0.980 / 0.47).
               With this script (official metrics at 512): step 540 img_roc_auc
               0.983 / pxl_avg_prec 0.554 (0.97-0.98 from 300 steps on).
               Below, the history of the sweep on the merged dataset (23k bees).
Training:      frozen ImageNet resnet18 teacher, resnet18 student with random
               weights trained to reproduce the teacher's layer1-3 features
               on healthy bees (VarroaDataset, TRAIN). The paper's optimiser
               (SGD lr 0.4, momentum 0.9, weight decay 1e-4, batch 32)
               with per-step cosine annealing, bf16 autocast in training.
Optimisation (sweep: experiments/single_bee/sweep_stfpm.py, logs/stfpm_sweep/;
sweep metrics: pixel AP at 128 px):
  - native resolution 512 (the varroa has a diameter of ~33 px): after 1 epoch
    img AUROC ~0.90 at 256 (max 0.917 in 2.5 epochs) vs ~0.97-0.98 at 512;
  - map = sum of layer2 and layer3 (MAP_LAYERS) with Gaussian blur
    (MAP_SIGMA): layer1 is noisy. The paper's product of the 3 layers without blur
    stops at img AUROC 0.947 / pixel AP 0.20 while layer2+layer3 with
    blur gives 0.983 / 0.53. Sum and product of layer2-3 are equivalent for
    AUROC, the sum is better for pixel AP; top-k score and per-layer
    normalisation do not help. layer3 alone has slightly higher pixel AP
    (~0.58 vs ~0.55) but lower img AUROC (~0.976);
  - short training: the student soon generalises to the varroa too. The peak
    is between 1 and 1.5 epochs; with 8 epochs img AUROC drops to 0.93 and pixel AP to
    0.27 while the loss keeps decreasing. 2 epochs with cosine (0.982-0.985
    from 1 epoch on) are more stable than 4 (0.975-0.983), hence evaluation
    every 1/4 epoch;
  - these do not help: flip + rot90 in training (peak 0.980), layer4 as an extra
    student layer (0.982, and the best map is still layer2+layer3).
With this script (official metrics at 512): step 905 (1.25 epochs)
img_roc_auc 0.985 / pxl_avg_prec 0.553, and 0.981-0.985 from 3/4 epoch on.
Best:          every EVAL_INTERVAL steps (and at the end) metrics on the validation set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/stfpm/ (best.pth +
               config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee stfpm

For the continual experiment (domain shift, replay) see main_stfpm_clad.py.
"""

from moviad.datasets.varroa.paths import DATA_ROOT
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
from moviad.models.stfpm.stfpm import STFPM
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
MODEL_NAME    = "stfpm"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE       = (512, 512)

# layers at stride 4, 8 and 16; other backbones only for the ablation (ABL_BACKBONE, see ablation.py)
BACKBONE_LAYERS = {
    "resnet18":        ["layer1", "layer2", "layer3"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "mobilenet_v2":    ["features.3", "features.6", "features.13"],
}
BACKBONE      = ablation.backbone("resnet18")
LAYERS        = BACKBONE_LAYERS[BACKBONE]
IMG_SIZE      = (512, 512)   # native crop resolution
MAP_LAYERS    = [2]          # indices in LAYERS of the layers used for the map (layer3 only; merged: [1, 2])
MAP_AGGREGATION = "sum"      # the paper uses "prod" over all layers
MAP_SIGMA     = 8.0          # Gaussian blur of the map (px at 512) before the score
TRAIN_STEPS   = 600          # cosine over 600 steps: the peak is between 300 and 420 steps (merged: 1448)
BATCH_SIZE    = 32
LR            = 0.4          # as in the STFPM paper (SGD, momentum 0.9), per-step cosine down to 0
WEIGHT_DECAY  = 1e-4
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 60           # evaluate every N steps and always at the end
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


def build_model(device):
    # frozen pretrained teacher, student with random weights
    teacher = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    student = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=False)
    return STFPM(teacher, student, map_aggregation=MAP_AGGREGATION,
                 map_layers=MAP_LAYERS, map_sigma=MAP_SIGMA).to(device)


def model_state(model):
    # the teacher is the ImageNet resnet18: the student is enough
    return clean_state_dict(model.student.model.state_dict())


def load_model_state(model, state):
    model.student.model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_stfpm_single_bee():
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

    # SGD on the student parameters only, per-step cosine annealing
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    total_steps = TRAIN_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training STFPM single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epochs "
          f"| {len(train_loader)} batches/epoch | lr {LR} | evaluation every {EVAL_INTERVAL} steps")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            model.train()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                teacher_features, student_features = model(batch)
            loss = 0.0
            for t, s in zip(teacher_features, student_features):
                t = F.normalize(t.float(), dim=1)
                s = F.normalize(s.float(), dim=1)
                loss = loss + (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoch {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(model, val_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(model))
            if step >= total_steps:
                break

    print(f"\nBest STFPM (\"epoch\" = training step) — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_stfpm_single_bee()
