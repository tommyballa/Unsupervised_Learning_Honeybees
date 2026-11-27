"""
Inferenza, metriche e heatmap normalizzate per gli esperimenti su immagini
singole (es. crop della singola ape).

Le heatmap sono normalizzate GLOBALMENTE (stessa scala per tutte le immagini
di test) e ancorate alla soglia di decisione:
    v_min  -> 0.0 (blu)   livello di fondo delle immagini sane
    soglia -> 0.5         soglia pixel che massimizza l'F1
    v_max  -> 1.0 (rosso) anomalie più forti del test set
Così un'immagine sana resta blu e solo ciò che supera la soglia diventa
giallo/rosso (con la normalizzazione min/max per immagine, invece, ogni
immagine avrebbe sempre un punto rosso).

Con un validation set (evaluate_and_report(val_loader=...)) le soglie image e
pixel e la normalizzazione delle heatmap sono fissate sul validation set:
img_f1 / pxl_f1 del test sono calcolati a quelle soglie, mentre
img_f1_oracle / pxl_f1_oracle sono quelli alla soglia ottimale sul test stesso
(ottimistici, solo come riferimento).

Mappe, metriche pixel e heatmap sono calcolate alla risoluzione delle maschere
GT del dataset (gt_mask_size): ogni modello può lavorare alla sua risoluzione
di input, la mappa viene riportata a quella delle maschere con interpolazione
bilineare, così le metriche dei diversi modelli sono confrontabili.
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
    """Immagine originale da disco, RGB uint8, alla risoluzione (H, W) delle maschere."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != tuple(size_hw):
        img = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
    return img


# ──────────────────────────────────────────────────────────────────────
# Inferenza + metriche
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataloader, device, map_postprocess: Callable | None = None):
    """
    Esegue il modello in fp32 sul test set.

    Args:
        map_postprocess: funzione opzionale applicata alle anomaly maps (B,1,H,W)
            del batch, alla risoluzione del modello (es. smoothing gaussiano).

    Returns:
        anomaly maps (N,1,H,W), scores (N,), labels (N,), masks GT (N,1,H,W),
        con H, W = risoluzione delle maschere GT
    """
    model.eval()
    maps, scores, labels, masks = [], [], [], []
    for images, batch_labels, batch_masks, _ in tqdm(dataloader, desc="Inferenza"):
        batch_maps, batch_scores = model(images.to(device))
        batch_maps = batch_maps.float().reshape(images.shape[0], 1, *batch_maps.shape[-2:])
        if map_postprocess is not None:
            batch_maps = map_postprocess(batch_maps)
        if batch_maps.shape[-2:] != batch_masks.shape[-2:]:
            # mappa alla risoluzione delle maschere GT (es. crop nativi 512x512)
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
            gt, pred = masks.copy(), maps  # ProAuc modifica gt in-place
        try:
            results[metric.name] = float(metric.compute(gt, pred))
        except Exception as e:
            print(f"  {metric.name}: ERRORE — {e}")
    return results


def print_metrics(results, indent="    "):
    for k, v in results.items():
        print(f"{indent}{k}: {v:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Normalizzazione delle heatmap
# ──────────────────────────────────────────────────────────────────────
def f1_optimal_threshold(gt, pred):
    """Soglia che massimizza l'F1 (la stessa usata dalla metrica F1 del report)."""
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def f1_at_threshold(gt, pred, threshold, chunk=256):
    """F1 con una soglia fissata (es. presa dal validation set), a blocchi di righe."""
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
    Calcola i parametri di una normalizzazione GLOBALE, uguale per tutte le
    immagini di test, ancorata alla soglia di decisione pixel:
        v_min  = 1° percentile dei pixel delle immagini sane   -> 0.0
        soglia = soglia pixel F1-ottimale (come pxl_f1)        -> 0.5
        v_max  = 99.99° percentile di tutti i pixel di test    -> 1.0
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
        # nessuna maschera GT disponibile: soglia al 99.9° percentile delle immagini sane
        threshold = float(np.percentile(normal_pixel_pred, 99.9))
    v_max = float(np.percentile(pixel_pred, 99.99))

    # garantisce v_min < soglia < v_max
    if v_max <= threshold:
        v_max = max(float(pixel_pred.max()), threshold + 1e-6)
    if v_min >= threshold:
        v_min = min(float(pixel_pred.min()), threshold - 1e-6)

    return {"v_min": v_min, "threshold": threshold, "v_max": v_max}


def normalize_anomaly_map(amap, norm):
    """Mappa lineare a tratti: [v_min, soglia] -> [0, 0.5], [soglia, v_max] -> [0.5, 1]."""
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
    """Overlay della heatmap normalizzata (colormap jet, identica alla colorbar)."""
    heatmap = _JET_LUT[(amap_norm * 255).astype(np.uint8)]
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)


def save_heatmap_figure(img_rgb, gt_mask, amap, norm, title, save_path, raw_label="Raw anomaly map"):
    """
    Figura con: originale | ground truth | mappa grezza (scala della singola
    immagine) | heatmap normalizzata globalmente | overlay con contorno GT.
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
    """Istogramma degli anomaly score per immagini sane e anomale."""
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
    Salva le heatmap di un campione bilanciato. norm e image_threshold (es. dal
    validation set) se None sono calcolati su questi dati (F1 ottimale).
    """
    os.makedirs(output_dir, exist_ok=True)

    if norm is None:
        norm = compute_heatmap_normalization(maps, masks, maps[labels == 0])
    if image_threshold is None:
        image_threshold = f1_optimal_threshold(labels, scores)
    print(f"\nNormalizzazione heatmap: v_min={norm['v_min']:.4f} | "
          f"soglia={norm['threshold']:.4f} | v_max={norm['v_max']:.4f}")
    print(f"Soglia image-level: {image_threshold:.4f}")

    save_score_distribution(scores, labels, image_threshold,
                            os.path.join(output_dir, "score_distribution.png"))

    rng = np.random.RandomState(seed)
    selected = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        selected.extend(rng.choice(idx, size=min(num_samples // 2, len(idx)), replace=False))

    print(f"Salvataggio di {len(selected)} heatmap in '{output_dir}'...")
    for i in selected:
        img_path = (test_dataset.image_paths[i] if hasattr(test_dataset, "image_paths")
                    else test_dataset[i][3])
        status = "Varroa" if labels[i] == 1 else "Sana"
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
    Valutazione finale: inferenza, metriche (con AU-PRO), heatmap normalizzate
    globalmente, distribuzione degli score e results.json in output_dir
    (extra_info, es. epoca della best, viene salvato in results.json).
    Con val_loader soglie F1 e normalizzazione delle heatmap vengono dal
    validation set (vedi docstring del modulo).
    """
    os.makedirs(output_dir, exist_ok=True)
    norm = image_threshold = val_results = None
    if val_loader is not None:
        maps, scores, labels, masks = run_inference(model, val_loader, device, map_postprocess)
        val_results = compute_metrics(maps, scores, labels, masks, with_pro=False)
        norm = compute_heatmap_normalization(maps, masks, maps[labels == 0])
        image_threshold = f1_optimal_threshold(labels, scores)
        print("Validation set (soglie e normalizzazione):")
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
    print(f"RISULTATI METRICHE — {title}")
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
    print(f"Heatmap e risultati salvati in {output_dir}")
    return results
