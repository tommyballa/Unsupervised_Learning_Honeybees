"""
Valutazione degli esperimenti full-image durante il training (scelta della best).

La valutazione usa la sliding window e lo score immagine dello script (passati
come funzioni) su un sottoinsieme fisso del test set: le prime N immagini
pulite e le prime N con varroa sintetica, in ordine alfabetico. Le metriche
pixel sono calcolate su un pixel ogni pixel_stride per lato, come nei plot
finali di RD4AD.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from moviad.utilities.evaluation.report import compute_metrics


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_paths(directory):
    return sorted(str(p) for p in Path(directory).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def eval_items(clean_dir, varroa_dir, per_class=None):
    """[(path, label)] con le prime per_class immagini pulite (0) e con varroa (1); None = tutte."""
    clean, varroa = image_paths(clean_dir), image_paths(varroa_dir)
    if per_class is not None:
        clean, varroa = clean[:per_class], varroa[:per_class]
    return [(p, 0) for p in clean] + [(p, 1) for p in varroa]


def load_gt_mask(img_path, label, shape, masks_dir):
    """Maschera GT binaria (uint8): DSC_5286_syn.jpg -> DSC_5286_mask.png."""
    if label == 0:
        return np.zeros(shape, dtype=np.uint8)
    base = re.sub(r"_syn$", "", Path(img_path).stem)
    mask_path = Path(masks_dir) / f"{base}_mask.png"
    if not mask_path.exists():
        print(f"    ⚠ Maschera non trovata per {Path(img_path).name}, uso maschera vuota")
        return np.zeros(shape, dtype=np.uint8)
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if mask.shape != tuple(shape):
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def evaluate_full_images(infer_fn, score_fn, items, masks_dir, pixel_stride=4):
    """
    Args:
        infer_fn: path immagine -> anomaly map (H, W), come nella valutazione finale
        score_fn: anomaly map -> score immagine
        items: [(path, label)] da eval_items
    Returns:
        metriche img/pxl (AUROC, AvgPrec, F1), senza AU-PRO
    """
    labels, scores, pixel_pred, pixel_gt = [], [], [], []
    for img_path, label in tqdm(items, desc="Valutazione"):
        amap = np.asarray(infer_fn(img_path), dtype=np.float32)
        labels.append(label)
        scores.append(float(score_fn(amap)))
        gt = load_gt_mask(img_path, label, amap.shape, masks_dir)
        pixel_pred.append(amap[::pixel_stride, ::pixel_stride].ravel())
        pixel_gt.append(gt[::pixel_stride, ::pixel_stride].ravel())

    return compute_metrics(np.concatenate(pixel_pred), np.array(scores), np.array(labels),
                           np.concatenate(pixel_gt), with_pro=False)
