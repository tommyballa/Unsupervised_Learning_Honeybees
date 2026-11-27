"""
SuperSimpleNet - Anomaly detection on single bees (single bee crops).

Training:      normal images of single bees (VarroaDataset, TRAIN split).
               Anomalies are simulated in feature space
               (Gaussian noise masked with Perlin noise), as in the paper.
               The paper's AdamW (adaptor 1e-4, segdec 2e-4) with per-step cosine
               annealing, fp16 autocast in training.
Optimisation (sweep: experiments/single_bee/sweep_supersimplenet.py,
logs/supersimplenet_sweep/; sweep metrics: pixel AP at 128 px, img AUROC
from the maximum of the map):
  - the paper's configuration (256x256, layer2+layer3, classification head
    score, 30 epochs) stops at img AUROC 0.726 / pixel AP 0.098
    (checkpoints/single_bee/supersimplenet_baseline_256/);
  - native resolution 512 (the varroa has a diameter of ~32 px): at 256 img AUROC
    ~0.65 after 750 steps vs ~0.88 at 512. Upscaling the input to 768 does not
    help (0.93);
  - score = maximum of the map (CLS_HEAD = False): the classification head
    sees extended synthetic anomalies (Perlin) and does not generalise to the varroa
    (img AUROC 0.55-0.80 while the map maximum gives 0.91-0.94), and
    training only the segmentation head does not worsen the map. Mean of the
    top-k pixels and blur before the score do not help;
  - layer1 at its own resolution (128x128 features): with layer2+layer3 all
    variants stop at 0.93-0.94 (64x64 or 128x128 features) and the errors are
    the small varroa (AUROC 0.89 below 800 px of area, 0.99 above 1300);
    with layer1-3 at 128x128 0.952-0.956. layer1 reduced to 64x64 does not help (0.90);
  - noise: std 0.02 (the paper's 0.015 is equivalent but slower; 0.01 too
    slow; 0.05 saturates at 0.91-0.92, 0.15 does not work). Perlin threshold 0.6,
    per-channel feature standardisation and 1x1 pooling do not help;
  - map blur (MAP_SIGMA, used only for the map): sigma 8 px at 512 is
    the best for pixel AP (0.45 vs 0.26 without blur), 16 px is worse;
  - the learning rate must decrease: at constant lr the metrics fluctuate a lot
    from one evaluation to the next, the best comes in the final part of the cosine.
With this script (official metrics at 512): step 6000 img_roc_auc 0.952 /
img_avg_prec 0.769 / pxl_avg_prec 0.516.
Best:          every EVAL_INTERVAL steps (and at the end) metrics on the test set;
               the best configuration (img_roc_auc, ties broken by pxl_avg_prec)
               is saved in checkpoints/single_bee/supersimplenet/
               (best.pth + config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee supersimplenet
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "supersimplenet"
REPO_ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR       = DATA_ROOT + "/processed/train_single_bee_batch3+4/cleaned_bees"
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE         = (512, 512)

BACKBONE        = "wide_resnet50_2"
LAYERS          = ["layer1", "layer2", "layer3"]   # the paper uses layer2 + layer3
IMG_SIZE        = (512, 512)   # native crop resolution
FEATURE_UPSCALE = 1            # features at layer1 resolution (128x128); in the paper 2x the first layer
NOISE_STD       = 0.02         # std of the Gaussian noise of the synthetic anomalies (paper: 0.015)
CLS_HEAD        = False        # segmentation head only: score = maximum of the map
MAP_SIGMA       = 8.0          # Gaussian blur of the map (px at 512), not used for the score
TRAIN_STEPS     = 6000         # ~4 epochs with batch 16
BATCH_SIZE      = 16
EVAL_INTERVAL   = 1000         # evaluate every N steps (and always at the end)
AMP             = True         # fp16 autocast in training (backbone and segmentation head)
NUM_WORKERS     = 8
SEED            = 1

# Optimiser from the paper / anomalib (AdamW), with per-step cosine annealing down to 0
ADAPTOR_LR      = 1e-4
SEGDEC_LR       = 2e-4
WEIGHT_DECAY    = 1e-5
PERLIN_THRESHOLD = SuperSimpleNet.DEFAULT_PARAMETERS["perlin_threshold"]

NUM_HEATMAPS    = 20           # half healthy, half with varroa
RAW_LABEL       = "Raw map (logit)"
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR  = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


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
def dataset_args():
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    return SuperSimpleNet(
        feature_extractor=feature_extractor,
        perlin_threshold=PERLIN_THRESHOLD,
        stop_grad=SuperSimpleNet.DEFAULT_PARAMETERS["stop_grad"],
        noise_std=NOISE_STD,
        upscale=FEATURE_UPSCALE,
        map_sigma=MAP_SIGMA,
        cls_head=CLS_HEAD,
    ).to(device)


def model_state(model):
    # the frozen backbone is not part of the state dict: adaptor + segdec
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_supersimplenet_single_bee():
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

    train_dataset = AugmentedTrainDataset(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Optimizer, scheduler (per step) and loss ---
    optimizer = AdamW([
        {"params": model.adaptor.parameters(), "lr": ADAPTOR_LR},
        {"params": model.segdec.parameters(), "lr": SEGDEC_LR, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = SSNLoss()
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training SuperSimpleNet single-bee | {TRAIN_STEPS} steps "
          f"| {len(train_loader)} batches/epoch | evaluation every {EVAL_INTERVAL} steps")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < TRAIN_STEPS:
        for images in tqdm(train_loader, desc=f"step {step}/{TRAIN_STEPS}", leave=False):
            model.train()
            images = images.to(device, non_blocking=True)
            B, _, H, W = images.shape
            # all training images are normal: masks and labels set to zero,
            # anomalies are generated by the model in feature space
            masks = torch.zeros(B, 1, H, W, device=device)
            labels = torch.zeros(B, device=device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                anomaly_map, anomaly_score, masks, labels = model(images=images, masks=masks, labels=labels)
            loss = loss_fn(pred_map=anomaly_map.float(),
                           pred_score=None if anomaly_score is None else anomaly_score.float(),
                           target_mask=masks, target_label=labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
                print(f"Step {step}/{TRAIN_STEPS} (epoch {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(model, test_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(model))
            if step >= TRAIN_STEPS:
                break

    print(f"\nBest SuperSimpleNet (\"epoch\" = training step) — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_supersimplenet_single_bee()
