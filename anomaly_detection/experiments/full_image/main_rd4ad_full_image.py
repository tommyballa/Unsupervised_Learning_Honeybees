"""
RD4AD (Reverse Distillation) - Anomaly detection on full-frame images with many bees.

Training: random crops from the high-resolution training images.
Inference: sliding window over the whole image, with Hanning window blending
           of the overlaps.
Evaluation: image- and pixel-level metrics (AUROC, AvgPrec, F1).
Visualization: heatmaps normalized GLOBALLY (same scale for all the test
               images) and anchored to the decision threshold:
                 v_min     -> 0.0 (blue) background level of the clean images
                 threshold -> 0.5        pixel threshold that maximizes F1
                 v_max     -> 1.0 (red)  strongest anomalies in the test set
               The normalization needs statistics of the whole test set,
               so the maps are first computed and saved to disk
               (.npy float16) and then visualized in a second pass.

Best: every EVAL_INTERVAL epochs (and at the last one) metrics on the first
EVAL_IMAGES_PER_CLASS clean and varroa images of the test set; the
best configuration (img_roc_auc, then pxl_avg_prec on ties) is saved in
checkpoints/full_image/rd4ad/ (best.pth + config.json).
Heatmaps and metrics on the whole test set from the best:
    python experiments/plot_heatmaps.py full_image rd4ad
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import json
import os
import re
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from moviad.models.rd4ad.rd4ad import RD4AD, RD4ADTrainArgs
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

BACKBONE     = "wide_resnet50_2"
TILE_SIZE    = 512          # RD4AD requires multiples of 32
EPOCHS       = 100
BATCH_SIZE   = 8
SEED         = 1

TOP_K        = 100          # image score = mean of the top-k pixels of the map
PIXEL_STRIDE = 4            # pixel subsampling for metrics and normalization

EVAL_INTERVAL = 10           # evaluate every N epochs (and always at the last one) for the best
EVAL_IMAGES_PER_CLASS = 5    # first N clean + N varroa test images used to pick the best

MODEL_NAME = "rd4ad"
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


def load_gt_mask(img_path, label, shape):
    """Binary GT mask (uint8) with the same size as the map."""
    if label == 0:
        return np.zeros(shape, dtype=np.uint8)
    mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
    if mask_path is None:
        print(f"    ⚠ Mask not found for {os.path.basename(img_path)}, using an empty mask")
        return np.zeros(shape, dtype=np.uint8)
    gt_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if gt_mask.shape != shape:
        gt_mask = cv2.resize(gt_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt_mask


class FullImageCropDataset(Dataset):
    """
    Loads high-resolution images and returns random crops.
    Pre-loads all images into RAM during __init__ to avoid the
    I/O bottleneck (reading a 24MP image from disk at every iteration).
    """

    def __init__(self, root_dir, crop_size=512, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crops/image → {len(self)} samples/epoch")

        print("[Dataset] Pre-loading the original images into RAM...")
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot read: {path}")
            self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("[Dataset] Pre-loading complete!")

    def __len__(self):
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx):
        img = self.images[idx // self.crops_per_image]

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
    Slices the image into tiles, runs RD4AD on each one,
    and rebuilds the global heatmap with Hanning window blending
    (reduces the border artifacts typical of RD4AD maps).

    The image is first padded by reflection of window_size // 4 pixels:
    this way even the pixels on the real border fall inside a tile (where the
    Hanning weight is high) instead of on its border, where RD4AD produces
    artificially high values that would dominate score and normalization.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H_orig, W_orig, _ = img.shape

    pad = window_size // 4
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    H, W, _ = img_padded.shape

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
            # RD4AD returns (anomaly_maps [B, 1, H, W], anomaly_scores [B])
            batch_amaps, _ = model(batch_tensor)
            batch_amaps = batch_amaps.float().cpu().numpy()

        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])  # (window_size, window_size)
            full_anomaly_map[y:y + window_size, x:x + window_size] += amap * hanning_2d
            weight_map[y:y + window_size, x:x + window_size] += hanning_2d

    for y in y_coords:
        for x in x_coords:
            tile = img_padded[y:y + window_size, x:x + window_size]
            batch_tiles.append(normalize(tile))
            batch_coords.append((y, x))

            if len(batch_tiles) == batch_size:
                process_batch(batch_tiles, batch_coords)
                batch_tiles, batch_coords = [], []

    if batch_tiles:
        process_batch(batch_tiles, batch_coords)

    full_anomaly_map /= (weight_map + 1e-8)
    return img, full_anomaly_map[pad:pad + H_orig, pad:pad + W_orig]


def top_k_score(amap, k=TOP_K):
    """Mean of the top-k pixels: a score robust to single noise spikes."""
    flat = amap.ravel()
    if flat.size <= k:
        return float(flat.max())
    return float(np.mean(np.partition(flat, -k)[-k:]))


# ──────────────────────────────────────────────────────────────────────
# Heatmap normalization
# ──────────────────────────────────────────────────────────────────────
def f1_optimal_threshold(gt, pred):
    """Threshold that maximizes F1 (the same one used by the report's F1 metric)."""
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_heatmap_normalization(pixel_pred, pixel_gt, normal_pixel_pred, max_pixels=5_000_000):
    """
    Computes the parameters of a GLOBAL normalization, the same for all
    test images, anchored to the pixel decision threshold:
        v_min     = 1st percentile of the clean-image pixels    -> 0.0
        threshold = F1-optimal pixel threshold (as in pxl_f1)   -> 0.5
        v_max     = 99.99th percentile of all test pixels       -> 1.0
    """
    pixel_pred = np.asarray(pixel_pred, dtype=np.float32).ravel()
    pixel_gt = np.asarray(pixel_gt).ravel() > 0
    normal_pixel_pred = np.asarray(normal_pixel_pred, dtype=np.float32).ravel()

    step = max(1, pixel_pred.size // max_pixels)
    pixel_pred, pixel_gt = pixel_pred[::step], pixel_gt[::step]
    normal_pixel_pred = normal_pixel_pred[::max(1, normal_pixel_pred.size // max_pixels)]

    v_min = float(np.percentile(normal_pixel_pred, 1))
    if pixel_gt.any() and not pixel_gt.all():
        threshold = f1_optimal_threshold(pixel_gt, pixel_pred)
    else:
        # no GT mask available: threshold at the 99.9th percentile of the clean images
        threshold = float(np.percentile(normal_pixel_pred, 99.9))
    v_max = float(np.percentile(pixel_pred, 99.99))

    # ensures v_min < threshold < v_max
    if v_max <= threshold:
        v_max = max(float(pixel_pred.max()), threshold + 1e-6)
    if v_min >= threshold:
        v_min = min(float(pixel_pred.min()), threshold - 1e-6)

    return {"v_min": v_min, "threshold": threshold, "v_max": v_max}


def normalize_anomaly_map(amap, norm):
    """Piecewise-linear map: [v_min, threshold] -> [0, 0.5], [threshold, v_max] -> [0.5, 1]."""
    lo, thr, hi = norm["v_min"], norm["threshold"], norm["v_max"]
    amap = amap.astype(np.float32)
    out = np.where(
        amap <= thr,
        0.5 * (amap - lo) / (thr - lo),
        0.5 + 0.5 * (amap - thr) / (hi - thr),
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


_JET_LUT = (plt.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def make_overlay(img_rgb, amap_norm, alpha=0.5):
    """Overlay of the normalized heatmap (jet colormap, identical to the colorbar)."""
    heatmap = _JET_LUT[(amap_norm * 255).astype(np.uint8)]
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)


def _plot_row(fig, axes, img_rgb, gt_mask, amap, amap_norm, norm, row_label):
    """One row of the figure: original | GT | raw map | normalized heatmap | overlay."""
    has_gt = gt_mask.any()

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original — {row_label}")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    im_raw = axes[2].imshow(amap, cmap="jet")
    axes[2].set_title("Raw map\n(per-image scale)")
    fig.colorbar(im_raw, ax=axes[2], fraction=0.046, pad=0.04)

    im_norm = axes[3].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Normalized heatmap\n(global scale, threshold = 0.5)")
    cbar = fig.colorbar(im_norm, ax=axes[3], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels([
        f"{norm['v_min']:.3g}",
        f"threshold {norm['threshold']:.3g}",
        f"{norm['v_max']:.3g}",
    ])

    axes[4].imshow(make_overlay(img_rgb, amap_norm))
    if has_gt:
        axes[4].contour(gt_mask, levels=[0.5], colors="white", linewidths=1)
    axes[4].set_title("Overlay" + (" (white contour = GT)" if has_gt else ""))

    for ax in axes:
        ax.axis("off")


def save_full_image_heatmap(img_rgb, gt_mask, amap, norm, title, output_dir, img_name,
                            display_scale=0.25, zoom_size=TILE_SIZE):
    """
    Saves:
      - overlay_<name>.jpg: full-resolution overlay of the normalized heatmap
      - plot_<name>.png: row 1 = whole image (downscaled), row 2 = full-resolution
        zoom around the maximum of the map
    """
    amap_norm = normalize_anomaly_map(amap, norm)

    overlay = make_overlay(img_rgb, amap_norm)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{Path(img_name).stem}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    H, W = amap.shape
    small = (max(1, int(W * display_scale)), max(1, int(H * display_scale)))
    img_s = cv2.resize(img_rgb, small, interpolation=cv2.INTER_AREA)
    amap_s = cv2.resize(amap, small, interpolation=cv2.INTER_AREA)
    # INTER_AREA + threshold > 0 so as not to lose small varroa mites in the downscaled version
    gt_s = (cv2.resize(gt_mask.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    amap_norm_s = normalize_anomaly_map(amap_s, norm)

    cy, cx = np.unravel_index(np.argmax(amap), amap.shape)
    half = zoom_size // 2
    y0 = int(np.clip(cy - half, 0, max(0, H - zoom_size)))
    x0 = int(np.clip(cx - half, 0, max(0, W - zoom_size)))
    zoom = (slice(y0, y0 + zoom_size), slice(x0, x0 + zoom_size))

    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    _plot_row(fig, axes[0], img_s, gt_s, amap_s, amap_norm_s, norm, "whole image")
    axes[0][0].add_patch(plt.Rectangle(
        (x0 * display_scale, y0 * display_scale), zoom_size * display_scale, zoom_size * display_scale,
        fill=False, edgecolor="yellow", linewidth=1.5,
    ))
    _plot_row(fig, axes[1], img_rgb[zoom], gt_mask[zoom], amap[zoom], amap_norm[zoom], norm,
              f"zoom on the maximum ({cx}, {cy})")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_{Path(img_name).stem}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(scores, labels, image_threshold, save_path):
    """Anomaly scores for clean and varroa images."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 30) if scores.max() > scores.min() else 10
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Clean", color="tab:blue")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Varroa", color="tab:red")
    ax.axvline(image_threshold, color="black", linestyle="--", label=f"F1 threshold = {image_threshold:.3g}")
    ax.set_xlabel(f"Anomaly score (mean of top-{TOP_K} pixels)")
    ax.set_ylabel("Number of images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Test + Metrics + Visualization
# ──────────────────────────────────────────────────────────────────────
def evaluate_and_visualize(model, device, output_dir, window_size=512):
    """
    Step 1: sliding window on clean images and images with synthetic varroa,
            computation of scores and (subsampled) pixels, and saving of the
            maps to disk.
    Step 2: metrics, global normalization and saving of the heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    maps_dir = os.path.join(output_dir, "anomaly_maps")
    os.makedirs(maps_dir, exist_ok=True)

    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)
    print(f"\nClean images (label=0): {len(clean_paths)}")
    print(f"Varroa images (label=1): {len(varroa_paths)}")

    test_items = [(p, 0) for p in clean_paths] + [(p, 1) for p in varroa_paths]

    records = []
    all_image_labels, all_image_scores = [], []
    all_pixel_gt, all_pixel_pred, normal_pixel_pred = [], [], []

    # --- Step 1: inference ---
    for img_path, label in test_items:
        img_name = os.path.basename(img_path)
        print(f"  [{'varroa' if label else 'clean'}] {img_name}...")

        _, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=BATCH_SIZE,
        )
        gt_mask = load_gt_mask(img_path, label, amap.shape)

        score = top_k_score(amap)
        all_image_labels.append(label)
        all_image_scores.append(score)

        sub_pred = amap[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
        all_pixel_pred.append(sub_pred)
        all_pixel_gt.append(gt_mask[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel())
        if label == 0:
            normal_pixel_pred.append(sub_pred)

        map_path = os.path.join(maps_dir, f"{'varroa' if label else 'clean'}_{Path(img_path).stem}.npy")
        np.save(map_path, amap.astype(np.float16))
        records.append({"img_path": img_path, "label": label, "score": score, "map_path": map_path})

    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)
    normal_pixel_pred = np.concatenate(normal_pixel_pred) if normal_pixel_pred else all_pixel_pred

    # --- Metric computation ---
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.IMAGE),
        F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.PIXEL),
    ]

    print(f"\n{'='*60}")
    print("METRIC RESULTS — RD4AD Full Image")
    print(f"{'='*60}")

    results = {}
    for metric in metrics:
        try:
            if metric.level == MetricLvl.IMAGE:
                value = metric.compute(all_image_labels, all_image_scores)
            else:
                value = metric.compute(all_pixel_gt, all_pixel_pred)
            results[metric.name] = float(value)
            print(f"  {metric.name:>20s}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name:>20s}: ERROR — {e}")

    print(f"{'='*60}\n")

    # --- Step 2: global normalization + heatmaps ---
    norm = compute_heatmap_normalization(all_pixel_pred, all_pixel_gt, normal_pixel_pred)
    if 0 < all_image_labels.sum() < len(all_image_labels):
        image_threshold = f1_optimal_threshold(all_image_labels, all_image_scores)
    else:
        image_threshold = float(np.median(all_image_scores))
    print(f"Heatmap normalization: v_min={norm['v_min']:.4f} | "
          f"threshold={norm['threshold']:.4f} | v_max={norm['v_max']:.4f}")
    print(f"Image-level threshold (optimal F1): {image_threshold:.4f}")

    save_score_distribution(all_image_scores, all_image_labels, image_threshold,
                            os.path.join(output_dir, "score_distribution.png"))

    print("\n--- Saving normalized heatmaps ---")
    for rec in records:
        img_name = os.path.basename(rec["img_path"])
        print(f"  {img_name}...")
        img_rgb = cv2.cvtColor(cv2.imread(rec["img_path"]), cv2.COLOR_BGR2RGB)
        amap = np.load(rec["map_path"]).astype(np.float32)
        gt_mask = load_gt_mask(rec["img_path"], rec["label"], amap.shape)

        status = "Varroa" if rec["label"] == 1 else "Clean"
        pred = "Varroa" if rec["score"] >= image_threshold else "Clean"
        title = (f"{img_name} — GT: {status} | Pred: {pred} | "
                 f"score {rec['score']:.4f} (threshold {image_threshold:.4f})")
        save_full_image_heatmap(img_rgb, gt_mask, amap, norm, title, output_dir, img_name,
                                zoom_size=window_size)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "metrics": results,
            "heatmap_normalization": norm,
            "image_threshold": image_threshold,
            "image_scores": {os.path.basename(r["img_path"]): r["score"] for r in records},
        }, f, indent=2)

    print(f"\nHeatmaps saved to {output_dir} (raw maps in {maps_dir})")


# ──────────────────────────────────────────────────────────────────────
# Model, best and heatmaps (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    model = RD4AD(backbone_name=BACKBONE, input_size=(TILE_SIZE, TILE_SIZE)).to(device)
    # the encoder (teacher) is frozen: the optimizer only updates bottleneck and decoder
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


def infer_image(model, device, img_path):
    """Anomaly map of the whole image, as in evaluate_and_visualize."""
    _, amap = sliding_window_inference(model, img_path, device, window_size=TILE_SIZE,
                                       stride=TILE_SIZE // 2, batch_size=BATCH_SIZE)
    return amap


def image_score(amap):
    return top_k_score(amap)


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
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FullImageCropDataset(
        TRAIN_DIR,
        crop_size=TILE_SIZE,
        crops_per_image=10,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
    )

    # --- Training args ---
    training_args = RD4ADTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE)
    training_args.init_train(model)
    tracker = BestTracker(
        CHECKPOINT_DIR, script_config(globals()),
        selection_data=(f"test set: first {EVAL_IMAGES_PER_CLASS} clean images "
                        f"+ {EVAL_IMAGES_PER_CLASS} with varroa"),
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training RD4AD full-image | {EPOCHS} epochs | {len(train_dataloader)} batches/epoch")
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
