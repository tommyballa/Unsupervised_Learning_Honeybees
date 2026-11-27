"""
Dinomaly - Anomaly detection on full-frame images with many bees.

Training: random crops from the high-resolution training images.
Inference: sliding window over the whole image, blending the overlaps.
Evaluation: image- and pixel-level metrics (AUROC, AvgPrec, F1).

Best: every EVAL_INTERVAL epochs (and at the last one) metrics on the first
EVAL_IMAGES_PER_CLASS clean and varroa images of the test set; the
best configuration (img_roc_auc, then pxl_avg_prec on ties) is saved in
checkpoints/full_image/dinomaly/ (best.pth + config.json).
Heatmaps and metrics on the whole test set from the best:
    python experiments/plot_heatmaps.py full_image dinomaly
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

from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.full_image import eval_items, evaluate_full_images
from moviad.utilities.evaluation.report import print_metrics


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = DATA_ROOT + "/processed/train/images/original"
TEST_CLEAN_DIR  = DATA_ROOT + "/processed/test/images"
TEST_VARROA_DIR = DATA_ROOT + "/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = DATA_ROOT + "/processed/test/synthetic_varroa_output/masks"

# Dinomaly ViT-base uses patch 14, img_size must be a multiple of 14
TILE_SIZE  = 336
EPOCHS     = 100
BATCH_SIZE = 6
SEED       = 1

EVAL_INTERVAL = 10           # evaluate every N epochs (and always at the last one) for the best
EVAL_IMAGES_PER_CLASS = 5    # first N clean + N varroa test images used to pick the best

MODEL_NAME = "dinomaly"
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
    img_name = Path(img_path).stem  # e.g. "DSC_5286_syn"
    # Strip the _syn suffix to get the base name
    base_name = re.sub(r"_syn$", "", img_name)  # e.g. "DSC_5286"
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
    """

    def __init__(self, root_dir, crop_size=336, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crops/image → {len(self)} samples/epoch")
        
        # Pre-load all images into RAM (needs about ~6-7 GB of RAM)
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

        # Random crop
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
                             window_size=336, stride=168, batch_size=8):
    """
    Slices the image into tiles, runs Dinomaly on each one,
    and rebuilds the global heatmap by averaging the overlaps.
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
            # Dinomaly returns (anomaly_maps [B, H, W], anomaly_scores [B])
            batch_amaps, _ = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()

        for i, (y, x) in enumerate(coords):
            amap = batch_amaps[i]  # shape: (window_size, window_size)
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
def evaluate_and_visualize(model, device, output_dir, window_size=336):
    """
    Runs the test on clean images and images with synthetic varroa,
    computes image- and pixel-level metrics, and saves the heatmap overlays.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Metrics to compute
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.IMAGE),
        F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.PIXEL),
    ]

    # --- Collect the test images ---
    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)

    print(f"\nClean images (label=0): {len(clean_paths)}")
    print(f"Varroa images (label=1): {len(varroa_paths)}")

    # Metric accumulators
    all_image_labels = []    # 0 = healthy, 1 = varroa
    all_image_scores = []    # anomaly score per image
    all_pixel_gt = []        # flattened GT masks
    all_pixel_pred = []      # anomaly maps flattened

    # --- Inference on CLEAN images (label=0, no mask) ---
    print("\n--- Inference on clean images ---")
    for img_path in clean_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=4,
        )

        # Image-level
        all_image_labels.append(0)
        all_image_scores.append(float(amap.max()))

        # Pixel-level: all-zero mask (no anomaly)
        gt_mask = np.zeros_like(amap, dtype=np.float32)
        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        # Save overlay
        _save_overlay(img_rgb, amap, output_dir, img_name)

    # --- Inference on VARROA images (label=1, with mask) ---
    print("\n--- Inference on synthetic varroa images ---")
    for img_path in varroa_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=4,
        )

        # Image-level
        all_image_labels.append(1)
        all_image_scores.append(float(amap.max()))

        # Pixel-level: load the GT mask
        mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
        if mask_path:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            gt_mask = (gt_mask > 127).astype(np.float32)  # binarize 0/1
            # Make sure mask and map have the same size
            if gt_mask.shape != amap.shape:
                gt_mask = cv2.resize(gt_mask, (amap.shape[1], amap.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            print(f"    ⚠ Mask not found for {img_name}, using an empty mask")
            gt_mask = np.zeros_like(amap, dtype=np.float32)

        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        # Save overlay
        _save_overlay(img_rgb, amap, output_dir, img_name)

    # --- Metric computation ---
    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)

    print(f"\n{'='*60}")
    print("METRIC RESULTS — Dinomaly Full Image")
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
    model = Dinomaly(
        encoder_name="vit_base_patch14_reg4_dinov2",
        target_layers=[2, 3, 4, 5],
        fuse_layer_encoder=[[0, 1, 2, 3]],
        img_size=(TILE_SIZE, TILE_SIZE),
    )
    return model.to(device)


def model_state(model):
    # frozen DINOv2 encoder: bottleneck + decoder are enough
    return clean_state_dict(model.trainable.state_dict())


def load_model_state(model, state):
    model.trainable.load_state_dict(state)


def infer_image(model, device, img_path):
    """Anomaly map of the whole image, as in evaluate_and_visualize."""
    _, amap = sliding_window_inference(model, img_path, device, window_size=TILE_SIZE,
                                       stride=TILE_SIZE // 2, batch_size=4)
    return amap


def image_score(amap):
    return float(amap.max())


def evaluate_subset(model, device):
    """Metrics (no plots) on the first EVAL_IMAGES_PER_CLASS clean and varroa images."""
    model.eval()
    items = eval_items(TEST_CLEAN_DIR, TEST_VARROA_DIR, EVAL_IMAGES_PER_CLASS)
    with torch.no_grad():
        return evaluate_full_images(lambda p: infer_image(model, device, p), image_score,
                                    items, TEST_MASKS_DIR)


def plot_heatmaps(model, device):
    """Metrics and heatmaps on the whole test set."""
    print(f"\nTest with sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR, window_size=TILE_SIZE)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
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

    # --- Training args ---
    total_iters = EPOCHS * len(train_dataloader)
    training_args = DinomalyTrainArgs(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        total_iters=total_iters,
    )
    training_args.init_train(model)
    tracker = BestTracker(
        CHECKPOINT_DIR, script_config(globals()),
        selection_data=(f"test set: first {EVAL_IMAGES_PER_CLASS} clean images "
                        f"+ {EVAL_IMAGES_PER_CLASS} with varroa"),
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training Dinomaly full-image | {EPOCHS} epochs | {len(train_dataloader)} batches/epoch")
    print(f"total_iters = {total_iters}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_dataloader, training_args)
        print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            results = evaluate_subset(model, device)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest {MODEL_NAME} full-image — {tracker.summary()}")
    print(f"Heatmaps and metrics on the whole test set: python experiments/plot_heatmaps.py full_image {MODEL_NAME}")


if __name__ == "__main__":
    main()
