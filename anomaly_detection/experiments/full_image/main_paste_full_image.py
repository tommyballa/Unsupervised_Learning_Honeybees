"""
PaSTe - Anomaly detection on full-frame images with many bees.

Training: random crops from the high-resolution training images.
Inference: sliding window over the whole image, blending the overlaps.
Evaluation: image- and pixel-level metrics (AUROC, AvgPrec, F1).

Varroa-specific tuning:
  - wide_resnet50_2 backbone with shallow layers (layer1, layer2, layer3)
    to capture fine textural details such as the varroa body.
  - PaSTe bootstrap from layer 0 (the teacher's conv1): the student shares
    only the first convolution, so layer1 remains an anomaly detection
    layer and no spatial resolution is lost.
  - output_size equal to TILE_SIZE to preserve the pixel-level resolution
    of the anomaly map.

Fixes compared to the previous version:
  - ad_layers=[1, 2, 3] instead of [0, 1, 2]. In PasteBackbone index 0 is
    conv1 alone (7x7, no BN/ReLU): a linear map, architecturally identical in
    teacher and student, that behaves the same way on normal and anomalous
    images, so its error map carries no information about anomalies.
    Since the layer maps are multiplied together, that noise modulated
    the whole anomaly map; in addition the deepest layer was layer2,
    with little context.
  - Teacher IMAGENET1K_V1, as in main_stfpm_full_image.py.
  - 100 epochs with cosine annealing and lr 0.04, like STFPM. The PaSTe loss
    uses MSE with reduction="sum", so for the same lr the per-sample step
    does not depend on the batch size: with a constant lr of 0.01 and 50 epochs
    the student received about 4 times fewer updates than STFPM's.
  - Heatmaps normalized globally (not min/max per image), so clean images
    no longer show fake red "anomalies".

Best: every EVAL_INTERVAL epochs (and at the last one) metrics on the first
EVAL_IMAGES_PER_CLASS clean and varroa images of the test set; the
best configuration (img_roc_auc, then pxl_avg_prec on ties) is saved in
checkpoints/full_image/paste/ (best.pth + config.json).
Heatmaps and metrics on the whole test set from the best:
    python experiments/plot_heatmaps.py full_image paste
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

from moviad.models.paste.paste import PaSTe, PasteTrainArgs
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.full_image import eval_items, evaluate_full_images
from moviad.utilities.evaluation.report import print_metrics


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = DATA_ROOT + "/processed/train/images"
TEST_CLEAN_DIR  = DATA_ROOT + "/processed/test/images"
TEST_VARROA_DIR = DATA_ROOT + "/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = DATA_ROOT + "/processed/test/synthetic_varroa_output/masks"

TILE_SIZE  = 512
EPOCHS     = 100
BATCH_SIZE = 16
LR         = 0.04
SEED       = 1

# PasteBackbone layer indices for wide_resnet50_2:
#   0 = conv1 (raw 7x7 conv, no BN/ReLU)
#   1 = bn1 + relu + maxpool + layer1
#   2 = layer2    3 = layer3    4 = layer4
# Layer 0 must not be used for anomaly detection (see docstring).
AD_LAYERS       = [1, 2, 3]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"

TOP_K = 100

EVAL_INTERVAL = 10           # evaluate every N epochs (and always at the last one) for the best
EVAL_IMAGES_PER_CLASS = 5    # first N clean + N varroa test images used to pick the best

MODEL_NAME = "paste"
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


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def top_k_mean(amap, k=TOP_K):
    """Mean of the k highest pixels: a score robust to single noise spikes."""
    flat = amap.ravel()
    if flat.size <= k:
        return float(flat.max())
    return float(np.mean(np.partition(flat, -k)[-k:]))


class FullImageCropDataset(Dataset):
    """
    Loads high-resolution images and returns random crops.
    Since the server has plenty of free RAM, pre-loads all images into memory
    during __init__ to avoid the severe I/O bottleneck
    (reading a 24MP image from disk at every single iteration).
    """

    def __init__(self, root_dir, crop_size=512, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crops/image → {len(self)} samples/epoch")

        print("[Dataset] Pre-loading the original images into RAM (4000x6016)...")
        self.images = [read_rgb(path) for path in self.image_paths]
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
    Slices the image into tiles, runs PaSTe on each one,
    and rebuilds the global heatmap by averaging the overlaps.

    PaSTe in eval mode returns:
      score_maps  [B, 1, output_size_h, output_size_w]
      anomaly_scores  [B]
    where output_size is set in the constructor (= TILE_SIZE in our case).
    """
    img = read_rgb(image_path)
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

    hanning_1d = np.hanning(window_size + 2)[1:-1]
    hanning_2d = np.outer(hanning_1d, hanning_1d).astype(np.float32)

    batch_tiles = []
    batch_coords = []
    model.eval()

    def process_batch(tiles, coords):
        batch_tensor = torch.stack(tiles).to(device)
        with torch.no_grad():
            # PaSTe returns (score_maps [B, 1, H, W], anomaly_scores [B])
            batch_amaps, _ = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()

        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])  # (window_size, window_size)
            full_anomaly_map[y:y + window_size, x:x + window_size] += amap * hanning_2d
            weight_map[y:y + window_size, x:x + window_size] += hanning_2d

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
def load_gt_mask(img_path, shape):
    mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
    if mask_path is None:
        print(f"    ⚠ Mask not found for {os.path.basename(img_path)}, using an empty mask")
        return np.zeros(shape, dtype=np.uint8)
    gt_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if gt_mask.shape != shape:
        gt_mask = cv2.resize(gt_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt_mask


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

    image_labels = []
    image_scores = []
    amaps = []
    gt_masks = []

    for split_name, paths, label in [
        ("clean", clean_paths, 0),
        ("synthetic varroa", varroa_paths, 1),
    ]:
        print(f"\n--- Inference on {split_name} images ---")
        for img_path in paths:
            print(f"  {os.path.basename(img_path)}...")
            _, amap = sliding_window_inference(
                model, img_path, device,
                window_size=window_size,
                stride=window_size // 2,
                batch_size=8,
            )
            amap = cv2.GaussianBlur(amap, (21, 21), 0)

            image_labels.append(label)
            image_scores.append(top_k_mean(amap))
            amaps.append(amap)
            if label == 1:
                gt_masks.append(load_gt_mask(img_path, amap.shape))
            else:
                gt_masks.append(np.zeros(amap.shape, dtype=np.uint8))

    # --- Global normalization and overlay saving ---
    # Same scale for all images: with per-image min/max even
    # the clean ones always showed a red peak.
    # Subsample so as not to copy dozens of 24MP images into RAM.
    sample = np.concatenate([a.ravel()[::97] for a in amaps])
    vmin = np.percentile(sample, 1)
    vmax = np.percentile(sample, 99.9)
    print(f"\nGlobal heatmap normalization: P1={vmin:.6f}, P99.9={vmax:.6f}")

    for img_path, amap in zip(clean_paths + varroa_paths, amaps):
        _save_overlay(read_rgb(img_path), amap, output_dir,
                      os.path.basename(img_path), vmin, vmax)

    # --- Metric computation ---
    image_labels = np.array(image_labels)
    image_scores = np.array(image_scores)
    pixel_gt = np.concatenate([m.ravel() for m in gt_masks])
    del gt_masks
    pixel_pred = np.concatenate([a.ravel() for a in amaps])
    del amaps

    print(f"\n{'='*60}")
    print("METRIC RESULTS — PaSTe Full Image")
    print(f"{'='*60}")

    for metric in metrics:
        try:
            if metric.level == MetricLvl.IMAGE:
                value = metric.compute(image_labels, image_scores)
            else:
                value = metric.compute(pixel_gt, pixel_pred)
            print(f"  {metric.name:>20s}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name:>20s}: ERROR — {e}")

    print(f"{'='*60}\n")
    print(f"Heatmaps saved to {output_dir}")


def _save_overlay(img_rgb, amap, output_dir, img_name, vmin, vmax):
    """Saves the heatmap overlay on the original image."""
    amap_norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# Model, best and heatmaps (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    # Teacher and student share conv1 (bootstrap 0); the student starts
    # from the output of the teacher's conv1 and recomputes layer1, layer2, layer3,
    # which are the anomaly detection layers (varroa ~2-3mm → shallow layers
    # are needed). output_size = (TILE_SIZE, TILE_SIZE) to keep the
    # pixel-level resolution of the anomaly maps equal to the tile's.
    model = PaSTe(
        backbone_model_name="wide_resnet50_2",
        input_size=(TILE_SIZE, TILE_SIZE),
        output_size=(TILE_SIZE, TILE_SIZE),
        ad_layers=AD_LAYERS,
        weights=TEACHER_WEIGHTS,
        student_bootstrap_layer=BOOTSTRAP_LAYER,
    )
    return model.to(device)


def model_state(model):
    # the teacher is the ImageNet wide_resnet50_2: the student is enough
    return clean_state_dict(model.student.state_dict())


def load_model_state(model, state):
    model.student.load_state_dict(state)


def infer_image(model, device, img_path):
    """Anomaly map of the whole image, as in evaluate_and_visualize."""
    _, amap = sliding_window_inference(model, img_path, device, window_size=TILE_SIZE,
                                       stride=TILE_SIZE // 2, batch_size=8)
    return cv2.GaussianBlur(amap, (21, 21), 0)


def image_score(amap):
    return top_k_mean(amap)


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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
    training_args = PasteTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training_args.optimizer, T_max=EPOCHS)
    tracker = BestTracker(
        CHECKPOINT_DIR, script_config(globals()),
        selection_data=(f"test set: first {EVAL_IMAGES_PER_CLASS} clean images "
                        f"+ {EVAL_IMAGES_PER_CLASS} with varroa"),
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training PaSTe full-image | {EPOCHS} epochs | {len(train_dataloader)} batches/epoch")
    print(f"AD layers: {AD_LAYERS} | bootstrap: {BOOTSTRAP_LAYER} | teacher: {TEACHER_WEIGHTS}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = model.train_epoch(epoch, train_dataloader, training_args)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            results = evaluate_subset(model, device)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest {MODEL_NAME} full-image — {tracker.summary()}")
    print(f"Heatmaps and metrics on the whole test set: python experiments/plot_heatmaps.py full_image {MODEL_NAME}")


if __name__ == "__main__":
    main()
