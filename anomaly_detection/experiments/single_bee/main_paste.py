"""
PaSTe (Partially Shared Teacher-student) - Anomaly detection on single bees
(single bee crops).

Training:      frozen ImageNet resnet18 teacher; the student (random weights)
               starts from the output of the teacher's conv1 (bootstrap layer 0) and
               learns to reproduce the layer1-3 features on healthy bees
               (VarroaDataset, TRAIN). SGD lr 0.4 (momentum 0.9, weight decay
               1e-4) with per-step cosine annealing over TRAIN_STEPS steps, bf16
               autocast; the loss uses MSE with reduction="sum". Input 512x512
               (native crop resolution).
Optimisation (sweep: experiments/single_bee/sweep_paste.py, logs/paste_sweep/;
batch3+4 dataset: 68.5k training bees, test 5000 healthy + 3119 with varroa;
sweep metrics: img AUROC / pixel AP at 128 px, best map variant):
  - the student soon generalises to the varroa too: the metrics peak
    after a few hundred steps and then drop while the loss keeps
    decreasing. With resnet18 and lr 0.4 the peak is at ~250 steps (0.944 / 0.43) and
    after 2000 steps it drops to 0.906 / 0.28; with 400 steps and cosine it stays at
    0.939-0.945 / 0.42-0.47 from 250 steps on, hence 400-step training
    with evaluation every 50;
  - resnet18 instead of wide_resnet50_2: the previous recipe (wide_resnet50_2,
    lr 0.04, map from layer3 only) stops at 0.89-0.90 / 0.21-0.26 in 3000
    steps, and 0.90-0.92 with lr 0.4;
  - map = sum of layer2 and layer3 with Gaussian blur (MAP_SIGMA 16 px at
    512): layer1 is noisy (0.92 with all 3 layers), without blur or with a 32 px blur
    it loses (0.92). layer3 alone has slightly higher pixel AP (~0.45 vs
    ~0.42) but slightly lower img AUROC; per-layer normalisation, product
    and top-k score are equivalent;
  - these do not help: loss only on bee cells (0.92-0.93; zeroing the map
    on the black background changes nothing), lr 0.1 (0.934-0.940), bootstrap at
    layer1 (0.90-0.92), input upscaled to 768 (0.92) or 1024 (0.87). Without
    bootstrap (student from the image, i.e. STFPM) the result is the same
    (0.945-0.948): sharing conv1 costs no accuracy. On the previous
    dataset weight EMA and flip/rot90 did not help either.
With this script (official metrics at 512): step 400 img_roc_auc 0.932 /
pxl_avg_prec 0.404, and 0.931-0.932 / 0.40-0.42 from 250 steps on.
Best:          every EVAL_INTERVAL epochs (and at the last one) metrics on the validation set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/paste/ (best.pth +
               config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee paste
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
from moviad.models.paste.paste import PaSTe, PasteTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "paste"
REPO_ROOT       = Path(__file__).resolve().parents[2]
DATASET_DIR     = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR       = f"{DATASET_DIR}/train/normal"
VAL_DIR         = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR        = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE         = (512, 512)

# indices of the blocks at stride 4, 8 and 16 (AD) and 8, 16 (map); other backbones
# only for the ablation (ABL_BACKBONE, see ablation.py)
BACKBONE_LAYERS = {
    "resnet18":        ([1, 2, 3], [2, 3]),
    "wide_resnet50_2": ([1, 2, 3], [2, 3]),
    "mobilenet_v2":    ([3, 6, 13], [6, 13]),
}
BACKBONE        = ablation.backbone("resnet18")
AD_LAYERS, MAP_LAYERS = BACKBONE_LAYERS[BACKBONE]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"
IMG_SIZE        = (512, 512)    # native crop resolution
MAP_AGGREGATION = "sum"
MAP_SIGMA       = 16.0        # Gaussian blur of the map (px at 512) before the score
TRAIN_STEPS     = 400        # the peak is at ~250 steps: short training with cosine down to 0
BATCH_SIZE      = 32
LR              = 0.4
AMP             = True       # bf16 autocast in training
EVAL_INTERVAL   = 50         # evaluate every N steps (and always at the end)
NUM_WORKERS     = 8
SEED            = 1

NUM_HEATMAPS    = 20         # half healthy, half with varroa
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR  = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
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
    return PaSTe(
        backbone_model_name=BACKBONE,
        input_size=IMG_SIZE,
        output_size=IMG_SIZE,
        ad_layers=AD_LAYERS,
        weights=TEACHER_WEIGHTS,
        student_bootstrap_layer=BOOTSTRAP_LAYER,
        map_aggregation=MAP_AGGREGATION,
        map_layers=MAP_LAYERS,
        map_sigma=MAP_SIGMA,
    ).to(device)


def model_state(model):
    # the teacher is the ImageNet backbone: the student is enough
    return clean_state_dict(model.student.state_dict())


def load_model_state(model, state):
    model.student.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_step(model, images, optimizer, scheduler, device):
    """Like PaSTe.train_step, but with bf16 autocast and a per-step scheduler."""
    model.train()
    images = images.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and device.type == "cuda"):
        t_feat, s_feat = model(images)
    loss = 0.0
    for t, s in zip(t_feat, s_feat):
        t = F.normalize(t.float(), dim=1)
        s = F.normalize(s.float(), dim=1)
        loss = loss + (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()


def infinite(loader):
    while True:
        yield from loader


def train_paste_single_bee():
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

    # SGD (momentum 0.9, weight decay 1e-4) on the student parameters only
    training_args = PasteTrainArgs(epochs=1, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training_args.optimizer, T_max=TRAIN_STEPS)
    # "epoch" of the best checkpoint = training step
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training PaSTe single-bee | {TRAIN_STEPS} steps | {len(train_loader)} batches/epoch")
    print(f"{'='*60}\n")

    batches = infinite(train_loader)
    total, n = 0.0, 0
    for step in tqdm(range(1, TRAIN_STEPS + 1), desc="Training"):
        total += train_step(model, next(batches), training_args.optimizer, scheduler, device)
        n += 1

        if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
            print(f"\nStep {step}/{TRAIN_STEPS} — loss: {total / n:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")
            total, n = 0.0, 0
            maps, scores, labels, masks = run_inference(model, val_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(step, results, lambda: model_state(model))

    print(f"\nBest PaSTe (\"epoch\" = training step) — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_paste_single_bee()
