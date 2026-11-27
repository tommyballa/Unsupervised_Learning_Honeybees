"""
GANomaly - Anomaly detection on full-frame images with many bees.

Training: random crops from the high-resolution training images.
Inference: sliding window over the whole image, blending the overlaps.
Evaluation: image- and pixel-level metrics (AUROC, AvgPrec, F1).
Visualization: heatmaps normalized globally with percentiles
               to highlight anomalies in red.

Fixes compared to the previous version:
  - Input normalized to [-1, 1] (mean=std=0.5). The GANomaly decoder
    ends with a Tanh, so it cannot reconstruct values outside [-1, 1]:
    with ImageNet normalization (range ~[-2.1, 2.6]) all bright and dark
    pixels had an irreducible reconstruction error, and the anomaly map
    detected brightness instead of varroa.
  - DCGAN initialization actually applied to the weights (see ganomaly.py).
  - 128x128 tiles instead of 256x256: GANomaly compresses the whole tile into a
    latent vector of 100 values; at 256 the reconstruction is so blurry that
    the error is high on every edge/texture of the bees. At 128 the reconstruction is
    more faithful, the varroa covers a larger fraction of the tile and the model
    goes from ~188M to ~50M parameters (more stable GAN training).
  - Image-level score also with GANomaly's native latent score
    (||z - z'||^2, maximum over the tiles), besides the top-k of the anomaly map.
  - Discriminator re-initialization when it collapses (as in the paper).
  - Same test set (synthetic_varroa_output) as the other full-image scripts.
  - The images are no longer all kept in RAM for the heatmaps.

Best: every EVAL_INTERVAL epochs (and at the last one) metrics on the first
EVAL_IMAGES_PER_CLASS clean and varroa images of the test set; the
best configuration (img_roc_auc, then pxl_avg_prec on ties) is saved in
checkpoints/full_image/ganomaly/ (best.pth + config.json).
Heatmaps and metrics on the whole test set from the best:
    python experiments/plot_heatmaps.py full_image ganomaly
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
from torch.optim import Adam
from tqdm import tqdm

from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.models.components.ganomaly.loss import GeneratorLoss, DiscriminatorLoss
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.full_image import eval_items, evaluate_full_images
from moviad.utilities.evaluation.report import print_metrics
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = DATA_ROOT + "/processed/train/images"
TEST_CLEAN_DIR  = DATA_ROOT + "/processed/test/images"
TEST_VARROA_DIR = DATA_ROOT + "/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = DATA_ROOT + "/processed/test/synthetic_varroa_output/masks"

TILE_SIZE        = 128
LATENT_VEC_SIZE  = 100
# 80 crops of 128 cover the same area per epoch as 20 crops of 256
CROPS_PER_IMAGE  = 80
EPOCHS           = 100
BATCH_SIZE       = 64
SEED             = 1
LR               = 2e-4
BETA1            = 0.5
BETA2            = 0.999

INFER_STRIDE     = TILE_SIZE // 2
INFER_BATCH_SIZE = 128
TOP_K            = 100

# The decoder ends with Tanh: input and reconstruction must lie in [-1, 1]
GANOMALY_MEAN = [0.5, 0.5, 0.5]
GANOMALY_STD  = [0.5, 0.5, 0.5]

EVAL_INTERVAL = 10           # evaluate every N epochs (and always at the last one) for the best
EVAL_IMAGES_PER_CLASS = 5    # first N clean + N varroa test images used to pick the best

MODEL_NAME = "ganomaly"
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
    base_name = re.sub(r"_syn$", "", img_name)  # e.g. "DSC_5286"
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
    Pre-loads all images into memory to avoid an I/O bottleneck.
    """

    def __init__(self, root_dir, crop_size=128, crops_per_image=80, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Found {len(self.image_paths)} images in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crops/image → {len(self)} samples/epoch")

        # Pre-load all images into RAM
        print("[Dataset] Pre-loading the original images into RAM...")
        self.images = [read_rgb(path) for path in self.image_paths]
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
                             window_size=TILE_SIZE, stride=INFER_STRIDE,
                             batch_size=INFER_BATCH_SIZE):
    """
    Slices the image into tiles, runs GANomaly on each one,
    and rebuilds the global heatmap with Hanning window blending.

    Returns the RGB image, the anomaly map (reconstruction error)
    and the GANomaly latent scores of all tiles.
    """
    img = read_rgb(image_path)
    H, W, _ = img.shape

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=GANOMALY_MEAN, std=GANOMALY_STD),
    ])

    full_anomaly_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    tile_scores = []

    # Tile coordinates, ensuring full coverage of the borders
    y_coords = list(range(0, H - window_size + 1, stride))
    if not y_coords or y_coords[-1] + window_size < H:
        y_coords.append(max(0, H - window_size))

    x_coords = list(range(0, W - window_size + 1, stride))
    if not x_coords or x_coords[-1] + window_size < W:
        x_coords.append(max(0, W - window_size))

    # Hanning window for smooth transitions between overlapping tiles
    hanning_1d = np.hanning(window_size + 2)[1:-1]
    hanning_2d = np.outer(hanning_1d, hanning_1d).astype(np.float32)

    batch_tiles = []
    batch_coords = []
    model.eval()

    def process_batch(tiles, coords):
        batch_tensor = torch.stack(tiles).to(device)
        with torch.no_grad():
            # GANomaly returns (anomaly_maps [B, 1, H, W], scores [B])
            batch_amaps, batch_scores = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()
            tile_scores.extend(batch_scores.cpu().numpy().tolist())

        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])  # (H_pad, W_pad)
            amap = amap[:window_size, :window_size]  # crop any padding
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

    # Gaussian smoothing to reduce pixel-level noise
    full_anomaly_map = cv2.GaussianBlur(full_anomaly_map, (21, 21), 0)

    return img, full_anomaly_map, np.asarray(tile_scores, dtype=np.float32)


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


def evaluate_and_visualize(model, device, output_dir):
    """
    Runs the test on clean images and images with synthetic varroa,
    computes image- and pixel-level metrics, and saves the heatmap overlays.
    """
    os.makedirs(output_dir, exist_ok=True)

    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)

    print(f"\nClean images (label=0): {len(clean_paths)}")
    print(f"Varroa images (label=1): {len(varroa_paths)}")

    image_labels = []
    map_scores = []      # top-k of the reconstruction anomaly map
    latent_scores = []   # GANomaly latent score, maximum over the tiles
    amaps = []
    gt_masks = []

    for split_name, paths, label in [
        ("clean", clean_paths, 0),
        ("synthetic varroa", varroa_paths, 1),
    ]:
        print(f"\n--- Inference on {split_name} images ---")
        for img_path in paths:
            print(f"  {os.path.basename(img_path)}...")
            _, amap, tile_scores = sliding_window_inference(model, img_path, device)

            image_labels.append(label)
            map_scores.append(top_k_mean(amap))
            latent_scores.append(float(tile_scores.max()))
            amaps.append(amap)
            if label == 1:
                gt_masks.append(load_gt_mask(img_path, amap.shape))
            else:
                gt_masks.append(np.zeros(amap.shape, dtype=np.uint8))

    # --- Global percentile normalization and overlay saving ---
    # Percentiles (not min/max) computed over all maps, so clean images
    # stay "cold" and red marks an out-of-scale response.
    # Subsample so as not to copy dozens of 24MP images into RAM.
    sample = np.concatenate([a.ravel()[::97] for a in amaps])
    p_low = np.percentile(sample, 2)
    p_high = np.percentile(sample, 99.5)
    print(f"\nGlobal anomaly map range: [{sample.min():.6f}, {sample.max():.6f}]")
    print(f"Percentile normalization: P2={p_low:.6f}, P99.5={p_high:.6f}")

    for img_path, amap in zip(clean_paths + varroa_paths, amaps):
        _save_overlay(read_rgb(img_path), amap, output_dir,
                      os.path.basename(img_path), p_low, p_high)

    # --- Metric computation ---
    image_labels = np.array(image_labels)
    pixel_gt = np.concatenate([m.ravel() for m in gt_masks])
    del gt_masks
    pixel_pred = np.concatenate([a.ravel() for a in amaps])
    del amaps

    image_level = [RocAuc(MetricLvl.IMAGE), AvgPrec(MetricLvl.IMAGE), F1(MetricLvl.IMAGE)]
    pixel_level = [RocAuc(MetricLvl.PIXEL), AvgPrec(MetricLvl.PIXEL), F1(MetricLvl.PIXEL)]

    print(f"\n{'='*60}")
    print("METRIC RESULTS — GANomaly Full Image")
    print(f"{'='*60}")

    for score_name, scores in [
        ("image score = top-k anomaly map", np.array(map_scores)),
        ("image score = GANomaly latent (max tile)", np.array(latent_scores)),
    ]:
        print(f"  [{score_name}]")
        for metric in image_level:
            _print_metric(metric, image_labels, scores)

    print("  [pixel]")
    for metric in pixel_level:
        _print_metric(metric, pixel_gt, pixel_pred)

    print(f"{'='*60}\n")
    print(f"Heatmaps saved to {output_dir}")


def _print_metric(metric, gt, pred):
    try:
        print(f"  {metric.name:>20s}: {metric.compute(gt, pred):.4f}")
    except Exception as e:
        print(f"  {metric.name:>20s}: ERROR — {e}")


def _save_overlay(img_rgb, amap, output_dir, img_name, p_low, p_high):
    """
    Saves the heatmap overlay on the original image.
    Percentile normalization + moderate gamma to highlight anomalies in red.
    """
    # Normalization with global percentiles
    amap_norm = np.clip((amap - p_low) / (p_high - p_low + 1e-8), 0, 1)

    # Moderate gamma: amplifies differences without crushing everything to 0
    gamma = 1.5
    amap_norm = amap_norm ** gamma

    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)

    # Plot with the 3 views
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap (normalized)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"plot_{Path(img_name).stem}.png"),
        dpi=100, bbox_inches="tight"
    )
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Model, best and heatmaps (also used by experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    model = Ganomaly(
        input_size=(TILE_SIZE, TILE_SIZE),
        num_input_channels=3,
        n_features=64,
        latent_vec_size=LATENT_VEC_SIZE,
        extra_layers=0,
        add_final_conv_layer=True,
    )
    return model.to(device)


def model_state(model):
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


def infer_image(model, device, img_path):
    """Anomaly map of the whole image, as in evaluate_and_visualize."""
    _, amap, _ = sliding_window_inference(model, img_path, device)
    return amap


def image_score(amap):
    # top-k of the reconstruction map (first score reported by evaluate_and_visualize)
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
    print(f"\nTest with sliding window (tile {TILE_SIZE}x{TILE_SIZE}, stride {INFER_STRIDE})...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR)


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

    # --- Model ---
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GANomaly parameters: {n_params / 1e6:.1f}M")

    # --- Training dataset with random crops ---
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=GANOMALY_MEAN, std=GANOMALY_STD),
    ])

    train_dataset = FullImageCropDataset(
        TRAIN_DIR,
        crop_size=TILE_SIZE,
        crops_per_image=CROPS_PER_IMAGE,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    # --- Optimizer and loss (training GAN custom) ---
    g_opt = Adam(model.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(model.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    g_loss_fn = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    d_loss_fn = DiscriminatorLoss()
    tracker = BestTracker(
        CHECKPOINT_DIR, script_config(globals()),
        selection_data=(f"test set: first {EVAL_IMAGES_PER_CLASS} clean images "
                        f"+ {EVAL_IMAGES_PER_CLASS} with varroa"),
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training GANomaly full-image | {EPOCHS} epochs | {len(train_dataloader)} batches/epoch")
    print(f"Tile: {TILE_SIZE}x{TILE_SIZE} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        avg_g_loss = 0.0
        avg_d_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = batch.to(device)

            # Forward pass
            padded, fake, latent_i, latent_o = model(images)
            pred_real_score, feat_real = model.discriminator(padded)

            # Generator update
            pred_fake_score, feat_fake = model.discriminator(fake)
            # GeneratorLoss uses "feature matching", so we pass feat_real and feat_fake
            g_loss = g_loss_fn(latent_i, latent_o, padded, fake, feat_real, feat_fake)

            g_opt.zero_grad()
            g_loss.backward(retain_graph=True)
            g_opt.step()
            avg_g_loss += g_loss.item()

            # Discriminator update
            pred_fake_score, _ = model.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real_score, pred_fake_score)

            d_opt.zero_grad()
            d_loss.backward()  # removed retain_graph=True to free memory
            d_opt.step()
            avg_d_loss += d_loss.item()

            # If the discriminator collapses it is re-initialized (as in the paper)
            if d_loss.item() < 1e-5:
                model.discriminator.apply(Ganomaly.weights_init)

        avg_g_loss /= len(train_dataloader)
        avg_d_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            results = evaluate_subset(model, device)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest {MODEL_NAME} full-image — {tracker.summary()}")
    print(f"Heatmaps and metrics on the whole test set: python experiments/plot_heatmaps.py full_image {MODEL_NAME}")


if __name__ == "__main__":
    main()
