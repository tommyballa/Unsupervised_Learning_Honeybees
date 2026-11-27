"""
PaDiM - Anomaly detection on single bees (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k healthy bees;
               val: 322 with varroa + 1251 healthy, for the best checkpoint and the F1 thresholds;
               test_heldout: 957 with varroa + 5000 healthy, evaluated once;
               see make_val_split.py).
Training:      features from a frozen ImageNet backbone (layers concatenated at the
               resolution of the first one, random reduction to PROJ_DIM channels as in the
               paper) of healthy bees (VarroaDataset, TRAIN), estimation of a
               multivariate Gaussian (covariance + COV_EPS I, 0.01 in the paper).
               Mean and covariance are accumulated in streaming on GPU (sums and
               outer products in float64), so all training images are used
               without keeping their features in RAM.
               GAUSSIAN = "position": one Gaussian per grid position
               (original PaDiM); "shared": a single Gaussian for all bee
               patches (the bees are not aligned, position is not informative).
               No epoch to choose: mean and covariance are
               saved in checkpoints/single_bee/padim/ (best.pth + config.json).
Evaluation:    per-patch Mahalanobis distance, map resized back to GT_SIZE,
               Gaussian blur MAP_SIGMA px (at GT_SIZE), zeroed outside the bee; image
               score = maximum of the map.
Optimisation (sweep: experiments/single_bee/sweep_padim.py, final dataset,
20k training images, image AUROC, pixel AP at 128 px; logs/padim_sweep/):
  - original PaDiM (wide_resnet50_2 layer1-3, 550 random channels, per-position
    Gaussian, eps 0.01, 256 px): 0.65 / 0.02. The per-position Gaussian
    is always worse (the bees are not aligned): with layer2-3 at 512 px 0.78
    vs 0.89 for the shared Gaussian;
  - eps 0.01 squashes the Mahalanobis towards a Euclidean distance: 1e-4 is the
    best almost everywhere (layer2-3 512 px shared, all channels: 0.937 -> 0.962);
  - 512 px instead of 256: 0.70 -> 0.89; 768 px is worse (0.96);
  - all channels instead of random projection to 550: 0.89 -> 0.962;
  - layer1 ruins everything (layer1-2: 0.54, layer2 alone 0.59); layer3
    only: 0.983 / 0.56; resnet18 layer2-3: 0.94;
  - 3x3 avg pool of the features: layer2-3 0.985, layer3 0.989 / 0.565 (chosen:
    half the channels and half the time of layer2-3 with pool 5, 0.989 / 0.562);
  - map blur: sigma 0-4 equivalent (0.989 / 0.988), 16 is worse.
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO):
    python experiments/plot_heatmaps.py single_bee padim
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "padim"
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATASET_DIR  = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR    = f"{DATASET_DIR}/train/normal"
VAL_DIR      = f"{DATASET_DIR}/val"           # best-checkpoint selection and F1 thresholds
TEST_DIR     = f"{DATASET_DIR}/test_heldout"  # evaluated only once on the best checkpoint
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE      = (512, 512)

# layers at stride 16; other backbones only for the ablation (ABL_BACKBONE, see ablation.py)
BACKBONE_LAYERS = {
    "wide_resnet50_2": ["layer3"],
    "resnet18":        ["layer3"],
    "mobilenet_v2":    ["features.13"],
}
BACKBONE     = ablation.backbone("wide_resnet50_2")
LAYERS       = BACKBONE_LAYERS[BACKBONE]
PROJ_DIM     = 0            # channels kept by the random projection (0 = all)
GAUSSIAN     = "shared"     # "position" (paper) | "shared"
FEAT_POOL    = 3            # local avg pool of the features (1 = none)
COV_EPS      = 1e-4         # 0.01 in the paper: too much for features with variance ~0.02
MAP_SIGMA    = 4            # px at GT_SIZE, Gaussian blur of the map
# bee mask: non-black pixels (the crop background is black, ~86% of the image),
# closing to fill the dark areas of the body, then a margin around the bee
FG_THRESHOLD = 10 / 255
FG_CLOSING   = 15           # px (kernel side)
FG_DILATION  = 8            # px of margin
IMG_SIZE     = (512, 512)
TRAIN_IMAGES = 0            # random subset of the training set (0 = all)
BATCH_SIZE   = 32
TEST_BATCH_SIZE = 32
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # half healthy, half with varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
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
    mask = 1 - _dilate(1 - _dilate(mask, FG_CLOSING), FG_CLOSING)    # closing
    return _dilate(mask, 2 * FG_DILATION + 1)


class PadimSingleBee(torch.nn.Module):
    """PaDiM with streaming estimation and a per-position or shared Gaussian."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
        with torch.no_grad():
            feats = self.feature_extractor(torch.zeros(1, 3, *IMG_SIZE, device=device))
        total_dim = sum(f.shape[1] for f in feats)
        self.grid = tuple(feats[0].shape[-2:])
        g = torch.Generator().manual_seed(SEED)
        dims = torch.randperm(total_dim, generator=g)[:PROJ_DIM] if PROJ_DIM else torch.arange(total_dim)
        self.register_buffer("dims", dims.sort().values.to(device))
        self.mean = None            # (C,) or (C, HW)
        self.cov = None             # (C, C) or (HW, C, C), without regularisation
        self.precision = None       # inverse of cov + COV_EPS I

    @torch.no_grad()
    def embed(self, images):
        """Embedding (B, C, h, w) at the resolution of the first layer."""
        feats = self.feature_extractor(images)
        feats = [f if f.shape[-2:] == self.grid else
                 F.interpolate(f, size=self.grid, mode="nearest") for f in feats]
        emb = torch.cat(feats, 1).index_select(1, self.dims)
        if FEAT_POOL > 1:
            emb = F.avg_pool2d(emb, FEAT_POOL, stride=1, padding=FEAT_POOL // 2)
        return emb.float()

    def grid_foreground(self, images):
        """Grid cells at least half covered by the bee (B, h*w)."""
        fg = F.adaptive_avg_pool2d(bee_foreground(images), self.grid)
        return (fg >= 0.5).flatten(1)

    @torch.no_grad()
    def fit(self, loader):
        n, s1, s2, shift = 0, None, None, None
        for batch in tqdm(loader, desc="Fitting PaDiM"):
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            emb = self.embed(images).flatten(2).double()            # (B, C, HW)
            if shift is None:
                shift = emb.mean(0) if GAUSSIAN == "position" else emb.mean((0, 2))
            if GAUSSIAN == "position":
                x = (emb - shift).permute(2, 0, 1)                    # (HW, B, C)
                if s1 is None:
                    s1 = torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
                    s2 = torch.zeros(x.shape[0], x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
                s1 += x.sum(1)
                s2.baddbmm_(x.transpose(1, 2), x)                     # in place: s2 is large
                n += x.shape[1]
            else:
                fg = self.grid_foreground(images)
                x = (emb - shift[None, :, None]).permute(0, 2, 1)[fg]  # (N_fg, C)
                s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
                s2 = x.T @ x if s2 is None else s2 + x.T @ x
                n += x.shape[0]
        self.set_statistics(n, s1, s2, shift)

    def set_statistics(self, n, s1, s2, shift):
        mu = s1 / n                                                  # (HW, C) or (C,)
        s2 -= n * mu[..., :, None] * mu[..., None, :]                # in place: s2 is large
        s2 /= n - 1
        self.cov = s2.float()
        if GAUSSIAN == "position":
            self.mean = (mu + shift.T).T.float().contiguous()          # (C, HW)
        else:
            self.mean = (mu + shift).float()
        self.set_precision(COV_EPS)

    def set_precision(self, eps):
        cov = self.cov.view(-1, *self.cov.shape[-2:])
        eye = torch.eye(cov.shape[-1], dtype=torch.float64, device=cov.device)
        precision = torch.empty_like(cov)
        for i in range(0, cov.shape[0], 256):                        # in blocks of positions
            c = cov[i:i + 256].double() + eps * eye
            precision[i:i + 256] = torch.cholesky_inverse(torch.linalg.cholesky(c)).float()
        self.precision = precision.view(self.cov.shape)

    @torch.no_grad()
    def distances(self, emb):
        """Per-patch Mahalanobis distance (B, 1, h, w)."""
        b, c, h, w = emb.shape
        if GAUSSIAN == "position":
            diff = (emb.flatten(2) - self.mean[None]).permute(2, 0, 1)    # (HW, B, C)
            d2 = ((diff @ self.precision) * diff).sum(-1).T                # (B, HW)
        else:
            diff = emb.flatten(2).transpose(1, 2) - self.mean              # (B, HW, C)
            d2 = ((diff @ self.precision) * diff).sum(-1)
        return d2.clamp_min(0).sqrt().reshape(b, 1, h, w)

    @torch.no_grad()
    def forward(self, images):
        dist = self.distances(self.embed(images))
        amap = F.interpolate(dist, size=GT_SIZE, mode="bilinear", align_corners=False)
        if MAP_SIGMA > 0:
            amap = gaussian_blur(amap, kernel_size=2 * int(4.0 * MAP_SIGMA + 0.5) + 1, sigma=MAP_SIGMA)
        amap = amap * bee_foreground(F.interpolate(images, size=GT_SIZE, mode="bilinear", align_corners=False))
        return amap, amap.flatten(1).amax(1)


def build_model(device):
    return PadimSingleBee(device)


def model_state(model):
    return {"mean": model.mean.cpu(), "cov": model.cov.cpu(), "dims": model.dims.cpu()}


def load_model_state(model, state):
    model.mean = state["mean"].to(model.device)
    model.cov = state["cov"].to(model.device)
    model.dims = state["dims"].to(model.device)
    model.set_precision(COV_EPS)


def train_dataset():
    dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    if TRAIN_IMAGES and TRAIN_IMAGES < len(dataset):
        subset = np.random.default_rng(SEED).choice(len(dataset), TRAIN_IMAGES, replace=False)
        dataset = torch.utils.data.Subset(dataset, sorted(subset.tolist()))
    return dataset


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_padim_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(device)
    dataset = train_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\nFitting PaDiM ({BACKBONE}, {LAYERS}, {len(model.dims)} channels, Gaussian {GAUSSIAN}, "
          f"grid {model.grid}) on {len(dataset)} healthy bees...")
    model.fit(loader)

    save_checkpoint(CHECKPOINT_DIR, model_state(model), script_config(globals()),
                    best_epoch=None, note="single-pass fitting (no epoch to choose)")
    print(f"PaDiM Gaussians saved to {CHECKPOINT_DIR}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_padim_single_bee()
