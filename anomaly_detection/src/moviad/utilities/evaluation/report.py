"""
Inference, metrics and normalized heatmaps for the experiments on single
images (e.g. single-bee crops).

Heatmaps are normalized GLOBALLY (same scale for all test images) and
anchored to the decision threshold:
    v_min     -> 0.0 (blue)  background level of the healthy images
    threshold -> 0.5         pixel threshold that maximizes F1
    v_max     -> 1.0 (red)   strongest anomalies in the test set
This way a healthy image stays blue and only what exceeds the threshold turns
yellow/red (with per-image min/max normalization, instead, every image
would always have a red spot).

With a validation set (evaluate_and_report(val_loader=...)) the image and
pixel thresholds and the heatmap normalization are fixed on the validation set:
the test img_f1 / pxl_f1 are computed at those thresholds, while
img_f1_oracle / pxl_f1_oracle are the ones at the optimal threshold on the test
set itself (optimistic, for reference only).

Maps, pixel metrics and heatmaps are computed at the resolution of the
dataset's GT masks (gt_mask_size): each model can work at its own input
resolution, and the map is resized to the mask resolution with bilinear
interpolation, so the metrics of different models are comparable.
"""

import json
import os
from typing import Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_rgb(img_path, size_hw):
    """Original image from disk, RGB uint8, at the (H, W) resolution of the masks."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != tuple(size_hw):
        img = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
    return img


# ──────────────────────────────────────────────────────────────────────
# Inference + metrics
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataloader, device, map_postprocess: Callable | None = None):
    """
    Runs the model in fp32 on the test set.

    Args:
        map_postprocess: optional function applied to the batch anomaly maps
            (B,1,H,W), at the model resolution (e.g. Gaussian smoothing).

    Returns:
        anomaly maps (N,1,H,W), scores (N,), labels (N,), GT masks (N,1,H,W),
        with H, W = resolution of the GT masks
    """
    model.eval()
    maps, scores, labels, masks = [], [], [], []
    for images, batch_labels, batch_masks, _ in tqdm(dataloader, desc="Inference"):
        batch_maps, batch_scores = model(images.to(device))
        batch_maps = batch_maps.float().reshape(images.shape[0], 1, *batch_maps.shape[-2:])
        if map_postprocess is not None:
            batch_maps = map_postprocess(batch_maps)
        if batch_maps.shape[-2:] != batch_masks.shape[-2:]:
            # map at the resolution of the GT masks (e.g. native 512x512 crops)
            batch_maps = F.interpolate(batch_maps, size=tuple(batch_masks.shape[-2:]),
                                       mode="bilinear", align_corners=False)
        maps.append(batch_maps.cpu().numpy())
        scores.append(batch_scores.float().cpu().numpy().reshape(-1))
        labels.append(np.asarray(batch_labels).reshape(-1))
        masks.append((batch_masks.cpu().numpy() > 0).astype(np.uint8))
    return np.concatenate(maps), np.concatenate(scores), np.concatenate(labels), np.concatenate(masks)


def compute_metrics(maps, scores, labels, masks, with_pro=True):
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.IMAGE),
        F1(MetricLvl.PIXEL),
    ]
    if with_pro:
        metrics.append(ProAuc(MetricLvl.PIXEL))

    results = {}
    for metric in metrics:
        if metric.level == MetricLvl.IMAGE:
            gt, pred = labels, scores
        else:
            gt, pred = masks.copy(), maps  # ProAuc modifies gt in place
        try:
            results[metric.name] = float(metric.compute(gt, pred))
        except Exception as e:
            print(f"  {metric.name}: ERROR — {e}")
    return results


def print_metrics(results, indent="    "):
    for k, v in results.items():
        print(f"{indent}{k}: {v:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Heatmap normalization
# ──────────────────────────────────────────────────────────────────────
def f1_optimal_threshold(gt, pred):
    """Threshold that maximizes F1 (the same one used by the report's F1 metric)."""
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def f1_at_threshold(gt, pred, threshold, chunk=256):
    """F1 at a fixed threshold (e.g. taken from the validation set), in row chunks."""
    tp = fp = fn = 0
    for i in range(0, len(pred), chunk):
        p = np.asarray(pred[i:i + chunk]) >= threshold
        g = np.asarray(gt[i:i + chunk]) > 0
        tp += int(np.count_nonzero(p & g))
        fp += int(np.count_nonzero(p & ~g))
        fn += int(np.count_nonzero(~p & g))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def compute_heatmap_normalization(pixel_pred, pixel_gt, normal_pixel_pred, max_pixels=5_000_000):
    """
    Computes the parameters of a GLOBAL normalization, the same for all
    test images, anchored to the pixel decision threshold:
        v_min     = 1st percentile of the healthy-image pixels   -> 0.0
        threshold = F1-optimal pixel threshold (as in pxl_f1)    -> 0.5
        v_max     = 99.99th percentile of all test pixels        -> 1.0
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
        # no GT mask available: threshold at the 99.9th percentile of the healthy images
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


def save_heatmap_figure(img_rgb, gt_mask, amap, norm, title, save_path, raw_label="Raw anomaly map"):
    """
    Figure with: original | ground truth | raw map (per-image
    scale) | globally normalized heatmap | overlay with GT contour.
    """
    amap_norm = normalize_anomaly_map(amap, norm)
    has_gt = gt_mask.any()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    im_raw = axes[2].imshow(amap, cmap="jet")
    axes[2].set_title(f"{raw_label}\n(per-image scale)")
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
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(scores, labels, image_threshold, save_path):
    """Histogram of the anomaly scores for healthy and anomalous images."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 60) if scores.max() > scores.min() else 10
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Healthy", color="tab:blue")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Varroa", color="tab:red")
    ax.axvline(image_threshold, color="black", linestyle="--", label=f"F1 threshold = {image_threshold:.3g}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Number of images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_anomaly_heatmaps(test_dataset, maps, scores, labels, masks, output_dir, num_samples=20,
                          seed=1, raw_label="Raw anomaly map", norm=None, image_threshold=None):
    """
    Saves the heatmaps of a balanced sample. norm and image_threshold (e.g. from
    the validation set), if None, are computed on these data (optimal F1).
    """
    os.makedirs(output_dir, exist_ok=True)

    if norm is None:
        norm = compute_heatmap_normalization(maps, masks, maps[labels == 0])
    if image_threshold is None:
        image_threshold = f1_optimal_threshold(labels, scores)
    print(f"\nHeatmap normalization: v_min={norm['v_min']:.4f} | "
          f"threshold={norm['threshold']:.4f} | v_max={norm['v_max']:.4f}")
    print(f"Image-level threshold: {image_threshold:.4f}")

    save_score_distribution(scores, labels, image_threshold,
                            os.path.join(output_dir, "score_distribution.png"))

    rng = np.random.RandomState(seed)
    selected = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        selected.extend(rng.choice(idx, size=min(num_samples // 2, len(idx)), replace=False))

    print(f"Saving {len(selected)} heatmaps to '{output_dir}'...")
    for i in selected:
        img_path = (test_dataset.image_paths[i] if hasattr(test_dataset, "image_paths")
                    else test_dataset[i][3])
        status = "Varroa" if labels[i] == 1 else "Healthy"
        gt_name = "Varroa" if labels[i] == 1 else "Healthy"
        pred = "Varroa" if scores[i] >= image_threshold else "Healthy"
        title = (f"GT: {gt_name} | Pred: {pred} | "
                 f"score {scores[i]:.4f} (threshold {image_threshold:.4f})")
        save_heatmap_figure(
            load_rgb(img_path, masks.shape[-2:]), masks[i, 0], maps[i, 0], norm, title,
            os.path.join(output_dir, f"sample_{status}_{i:05d}_score_{scores[i]:.4f}.png"),
            raw_label=raw_label,
        )

    return norm, image_threshold


def evaluate_and_report(model, test_dataset, test_loader, device, output_dir, title,
                        num_heatmaps=20, seed=1,
                        map_postprocess: Callable | None = None, raw_label="Raw anomaly map",
                        extra_info: dict | None = None, val_loader=None):
    """
    Final evaluation: inference, metrics (with AU-PRO), globally normalized
    heatmaps, score distribution and results.json in output_dir
    (extra_info, e.g. the best epoch, is saved in results.json).
    With val_loader, the F1 thresholds and the heatmap normalization come from
    the validation set (see the module docstring).
    """
    os.makedirs(output_dir, exist_ok=True)
    norm = image_threshold = val_results = None
    if val_loader is not None:
        maps, scores, labels, masks = run_inference(model, val_loader, device, map_postprocess)
        val_results = compute_metrics(maps, scores, labels, masks, with_pro=False)
        norm = compute_heatmap_normalization(maps, masks, maps[labels == 0])
        image_threshold = f1_optimal_threshold(labels, scores)
        print("Validation set (thresholds and normalization):")
        print_metrics(val_results)
        del maps, masks

    maps, scores, labels, masks = run_inference(model, test_loader, device, map_postprocess)
    results = compute_metrics(maps, scores, labels, masks)
    if val_loader is not None:
        results["img_f1_oracle"] = results.pop("img_f1")
        results["pxl_f1_oracle"] = results.pop("pxl_f1")
        results["img_f1"] = f1_at_threshold(labels, scores, image_threshold)
        results["pxl_f1"] = f1_at_threshold(masks, maps, norm["threshold"])

    print(f"\n{'='*60}")
    print(f"METRIC RESULTS — {title}")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:>20s}: {v:.4f}")
    print(f"{'='*60}")

    norm, image_threshold = save_anomaly_heatmaps(
        test_dataset, maps, scores, labels, masks, output_dir,
        num_samples=num_heatmaps, seed=seed, raw_label=raw_label,
        norm=norm, image_threshold=image_threshold,
    )
    summary = {"metrics": results, "heatmap_normalization": norm, "image_threshold": image_threshold,
               "thresholds_from": "validation set" if val_loader is not None else "test set"}
    if val_results is not None:
        summary["val_metrics"] = val_results
    if extra_info:
        summary["checkpoint"] = extra_info
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Heatmaps and results saved to {output_dir}")
    return results
