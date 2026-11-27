"""
PaDiM - Anomaly detection on full-frame images with many bees.

Training: random crops from the high-resolution training images.
  PaDiM does not use backprop: it collects the feature maps and fits
  a multivariate Gaussian distribution for each spatial position.
Inference: sliding window over the whole image, blending the overlaps.
Evaluation: image- and pixel-level metrics (AUROC, AvgPrec, F1).

Varroa-specific tuning:
  - wide_resnet50_2 backbone with layers (layer1, layer2, layer3)
    to capture fine textural details such as the varroa body.
  - Dimensional random projection (1792 → 550) to reduce memory
    without losing discriminative power.
  - Post-inference Gaussian smoothing to improve the localization
    of anomalies in areas with many bees close together.

The estimated Gaussians are not saved (at 512 px with wide_resnet50_2 the
covariance takes ~20 GB): this script and
    python experiments/plot_heatmaps.py full_image padim
redo the fitting (a single pass over the training crops) and generate metrics
and heatmaps on the whole test set.
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import os
import re
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from moviad.models.padim.padim import Padim, PadimTrainArgs
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = DATA_ROOT + "/processed/train/images"
TEST_CLEAN_DIR  = DATA_ROOT + "/processed/test/images"
TEST_VARROA_DIR = DATA_ROOT + "/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = DATA_ROOT + "/processed/test/synthetic_varroa_output/masks"

TILE_SIZE  = 512
BATCH_SIZE = 8
SEED       = 1

MODEL_NAME = "padim"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "full_image" / MODEL_NAME


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


def get_image_paths(directory):
    """Returns the sorted image paths."""
    return sorted([
        str(p) for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def find_mask_for_image(img_path, masks_dir):
    """
    Given a varroa image path (e.g. DSC_5286_syn.jpg),
    finds the matching mask (e.g. DSC_5286_mask.png).
    """
    img_name = Path(img_path).stem
    base_name = re.sub(r"_syn$", "", img_name)
    mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
    if os.path.exists(mask_path):
        return mask_path
    return None


class FullImageCropDataset(Dataset):
    """
    Loads high-resolution images and returns random crops.
    Since the server has plenty of free RAM, pre-loads all images into memory
    during __init__ to avoid the severe I/O bottleneck
    (reading a 24MP image from disk at every single iteration).

    For PaDiM: returns only the tensor (no label) because
    train_step() in padim.py handles the batch tensor directly.
    """

    def __init__(self, root_dir, crop_size=512, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crops/image → {len(self)} samples/epoch")

        print("[Dataset] Pre-loading the original images into RAM (4000x6016)...")
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot read: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
        print("[Dataset] Pre-loading complete!")

    def __len__(self):
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        img = self.images[img_idx]

        H, W, _ = img.shape
        y = random.randint(0, max(0, H - self.crop_size))
        x = random.randint(0, max(0, W - self.crop_size))
        crop = img[y:y + self.crop_size, x:x + self.crop_size]

        if self.transform:
            crop = self.transform(crop)
        return crop


# ──────────────────────────────────────────────────────────────────────
# Sliding window inference
# ──────────────────────────────────────────────────────────────────────
def sliding_window_inference(model, image_path, device,
                             window_size=512, stride=256, batch_size=8):
    """
    Slices the image into tiles, runs PaDiM on each one,
    and rebuilds the global heatmap by averaging the overlaps.

    PaDiM in eval mode returns:
      score_map     [B, 1, H, W]  (torch tensor, already on CPU)
      img_scores    [B]           (torch tensor, already on CPU)
    where H, W equal the input size (upsampled internally).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_anomaly_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Tile coordinates, ensuring full coverage of the borders
    y_coords = list(range(0, H - window_size + 1, stride))
    if not y_coords or y_coords[-1] + window_size < H:
        y_coords.append(max(0, H - window_size))

    x_coords = list(range(0, W - window_size + 1, stride))
    if not x_coords or x_coords[-1] + window_size < W:
        x_coords.append(max(0, W - window_size))

    batch_tiles = []
    batch_coords = []
    model.eval()

    def process_batch(tiles, coords):
        batch_tensor = torch.stack(tiles).to(device)
        with torch.no_grad():
            # PaDiM returns (score_map [B, 1, H, W], img_scores [B])
            # compute_distances and gaussian_filter run on CPU internally
            batch_amaps, _ = model(batch_tensor)
            # batch_amaps is a torch tensor from numpy, it may be on CPU
            if isinstance(batch_amaps, torch.Tensor):
                batch_amaps = batch_amaps.cpu().numpy()
            else:
                batch_amaps = np.array(batch_amaps)

        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])  # (window_size, window_size)
            full_anomaly_map[y:y + window_size, x:x + window_size] += amap
            weight_map[y:y + window_size, x:x + window_size] += 1.0

    for y in y_coords:
        for x in x_coords:
            tile = img[y:y + window_size, x:x + window_size]
            batch_tiles.append(normalize(tile))
            batch_coords.append((y, x))

            if len(batch_tiles) == batch_size:
                process_batch(batch_tiles, batch_coords)
                batch_tiles, batch_coords = [], []

    if batch_tiles:
        process_batch(batch_tiles, batch_coords)

    full_anomaly_map /= (weight_map + 1e-8)
    return img, full_anomaly_map


# ──────────────────────────────────────────────────────────────────────
# Test + Metrics + Visualization
# ──────────────────────────────────────────────────────────────────────
def evaluate_and_visualize(model, device, output_dir, window_size=512):
    """
    Runs the test on clean images and images with synthetic varroa,
    computes image- and pixel-level metrics, and saves the heatmap overlays.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        RocAuc(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.IMAGE),
        F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.PIXEL),
    ]

    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)

    print(f"\nClean images (label=0): {len(clean_paths)}")
    print(f"Varroa images (label=1): {len(varroa_paths)}")

    all_image_labels = []
    all_image_scores = []
    all_pixel_gt = []
    all_pixel_pred = []

    # --- Inference on CLEAN images (label=0) ---
    print("\n--- Inference on clean images ---")
    for img_path in clean_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=8,
        )

        all_image_labels.append(0)
        all_image_scores.append(float(amap.max()))

        gt_mask = np.zeros_like(amap, dtype=np.float32)
        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        _save_overlay(img_rgb, amap, output_dir, img_name)

    # --- Inference on VARROA images (label=1) ---
    print("\n--- Inference on synthetic varroa images ---")
    for img_path in varroa_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=8,
        )

        all_image_labels.append(1)
        all_image_scores.append(float(amap.max()))

        mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
        if mask_path:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            gt_mask = (gt_mask > 127).astype(np.float32)
            if gt_mask.shape != amap.shape:
                gt_mask = cv2.resize(gt_mask, (amap.shape[1], amap.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            print(f"    ⚠ Mask not found for {img_name}, using an empty mask")
            gt_mask = np.zeros_like(amap, dtype=np.float32)

        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        _save_overlay(img_rgb, amap, output_dir, img_name)

    # --- Metric computation ---
    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)

    print(f"\n{'='*60}")
    print("METRIC RESULTS — PaDiM Full Image")
    print(f"{'='*60}")

    for metric in metrics:
        try:
            if metric.level == MetricLvl.IMAGE:
                value = metric.compute(all_image_labels, all_image_scores)
            else:
                value = metric.compute(all_pixel_gt, all_pixel_pred)
            print(f"  {metric.name:>20s}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name:>20s}: ERROR — {e}")

    print(f"{'='*60}\n")
    print(f"Heatmaps saved to {output_dir}")


def _save_overlay(img_rgb, amap, output_dir, img_name):
    """Saves the heatmap overlay on the original image."""
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# Model, best and heatmaps (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    # wide_resnet50_2 with layers (layer1, layer2, layer3):
    #   total embedding = 1792 channels → reduced to 550 by random projection.
    # These layers capture both low-frequency patterns (bee shape)
    # and high-frequency textures (varroa, legs, wings), essential to
    # tell anomalies apart in crowded images.
    model = Padim(
        backbone_model_name="wide_resnet50_2",
        layers_idxs=(1, 2, 3),
        device=device,
        class_name="bees_varroa",
    )
    model.to(device)
    return model


def fit_model(device):
    """Builds PaDiM and estimates the Gaussians on the training crops (no epochs)."""
    model = build_model(device)

    # --- Training dataset with random crops ---
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    train_dataset = FullImageCropDataset(
        TRAIN_DIR,
        crop_size=TILE_SIZE,
        crops_per_image=10,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )

    # --- Training (single epoch: PaDiM collects statistics without backprop) ---
    training_args = PadimTrainArgs(epochs=1, batch_size=BATCH_SIZE)
    training_args.init_train(model)

    print(f"\n{'='*60}")
    print(f"Training PaDiM full-image | 1 epoch (statistical fitting) | {len(train_dataloader)} batches")
    print(f"{'='*60}\n")

    model.train()
    model.train_epoch(0, train_dataloader, training_args)
    print("Multivariate Gaussian fitting complete.")
    return model


def load_best(device):
    """For experiments/plot_heatmaps.py: the Gaussians are not saved, so the fitting is repeated."""
    return fit_model(device), {"note": "fitting repeated: the 512 px Gaussians (~20 GB) are not saved"}


def plot_heatmaps(model, device):
    """Metrics and heatmaps on the whole test set."""
    print(f"\nTest with sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR, window_size=TILE_SIZE)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = fit_model(device)
    plot_heatmaps(model, device)


if __name__ == "__main__":
    main()
