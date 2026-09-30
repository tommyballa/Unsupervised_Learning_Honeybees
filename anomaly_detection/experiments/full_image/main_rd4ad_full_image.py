"""
RD4AD (Reverse Distillation) - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con Hanning window blending
           delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).
Visualizzazione: heatmap normalizzate GLOBALMENTE (stessa scala per tutte le
                 immagini di test) e ancorate alla soglia di decisione:
                   v_min  -> 0.0 (blu)   livello di fondo delle immagini pulite
                   soglia -> 0.5         soglia pixel che massimizza l'F1
                   v_max  -> 1.0 (rosso) anomalie più forti del test set
                 La normalizzazione richiede le statistiche di tutto il test
                 set, quindi le mappe vengono prima calcolate e salvate su
                 disco (.npy float16) e poi visualizzate in un secondo passaggio.
"""

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


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/images"
TEST_CLEAN_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/images"
TEST_VARROA_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/masks"

BACKBONE     = "wide_resnet50_2"
TILE_SIZE    = 512          # RD4AD richiede multipli di 32
EPOCHS       = 100
BATCH_SIZE   = 8
SEED         = 1

TOP_K        = 100          # image score = media dei top-k pixel della mappa
PIXEL_STRIDE = 4            # sottocampionamento dei pixel per metriche e normalizzazione

REPO_ROOT    = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "full_image" / "rd4ad"
CHECKPOINT   = REPO_ROOT / "checkpoints" / "full_image" / "rd4ad.pth"


# ──────────────────────────────────────────────────────────────────────
# Utilità
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_image_paths(directory):
    """Restituisce i path delle immagini ordinati."""
    return sorted([
        str(p) for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def find_mask_for_image(img_path, masks_dir):
    """
    Dato un path immagine varroa (es. DSC_5286_syn.jpg),
    trova la maschera corrispondente (es. DSC_5286_mask.png).
    """
    img_name = Path(img_path).stem
    base_name = re.sub(r"_syn$", "", img_name)
    mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
    if os.path.exists(mask_path):
        return mask_path
    return None


def load_gt_mask(img_path, label, shape):
    """Maschera GT binaria (uint8) della stessa dimensione della mappa."""
    if label == 0:
        return np.zeros(shape, dtype=np.uint8)
    mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
    if mask_path is None:
        print(f"    ⚠ Maschera non trovata per {os.path.basename(img_path)}, uso maschera vuota")
        return np.zeros(shape, dtype=np.uint8)
    gt_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if gt_mask.shape != shape:
        gt_mask = cv2.resize(gt_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt_mask


class FullImageCropDataset(Dataset):
    """
    Carica immagini ad alta risoluzione e restituisce crop casuali.
    Pre-carica tutte le immagini in RAM durante l'__init__ per evitare il
    collo di bottiglia I/O (lettura da disco di 24MP a ogni iterazione).
    """

    def __init__(self, root_dir, crop_size=512, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Trovate {len(self.image_paths)} immagini in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crop/immagine → {len(self)} campioni/epoca")

        print("[Dataset] Pre-caricamento in RAM delle immagini originali...")
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Impossibile leggere: {path}")
            self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("[Dataset] Pre-caricamento completato!")

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
    Taglia l'immagine in tile, esegue RD4AD su ciascuna,
    e ricostruisce la heatmap globale con Hanning window blending
    (riduce gli artefatti di bordo tipici delle mappe di RD4AD).

    L'immagine viene prima estesa per riflessione di window_size // 4 pixel:
    così anche i pixel sul bordo reale cadono all'interno di un tile (dove il
    peso di Hanning è alto) invece che sul suo bordo, dove RD4AD produce
    valori artificialmente alti che dominerebbero score e normalizzazione.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile leggere: {image_path}")
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

    # Coordinate dei tile, assicurando copertura completa dei bordi
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
            # RD4AD restituisce (anomaly_maps [B, 1, H, W], anomaly_scores [B])
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
    """Media dei top-k pixel: score robusto a singoli spike di rumore."""
    flat = amap.ravel()
    if flat.size <= k:
        return float(flat.max())
    return float(np.mean(np.partition(flat, -k)[-k:]))


# ──────────────────────────────────────────────────────────────────────
# Normalizzazione delle heatmap
# ──────────────────────────────────────────────────────────────────────
def f1_optimal_threshold(gt, pred):
    """Soglia che massimizza l'F1 (la stessa usata dalla metrica F1 del report)."""
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def compute_heatmap_normalization(pixel_pred, pixel_gt, normal_pixel_pred, max_pixels=5_000_000):
    """
    Calcola i parametri di una normalizzazione GLOBALE, uguale per tutte le
    immagini di test, ancorata alla soglia di decisione pixel:
        v_min  = 1° percentile dei pixel delle immagini pulite -> 0.0
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
        # nessuna maschera GT disponibile: soglia al 99.9° percentile delle immagini pulite
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


def _plot_row(fig, axes, img_rgb, gt_mask, amap, amap_norm, norm, row_label):
    """Una riga della figura: originale | GT | mappa grezza | heatmap normalizzata | overlay."""
    has_gt = gt_mask.any()

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Originale — {row_label}")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    im_raw = axes[2].imshow(amap, cmap="jet")
    axes[2].set_title("Mappa grezza\n(scala della singola immagine)")
    fig.colorbar(im_raw, ax=axes[2], fraction=0.046, pad=0.04)

    im_norm = axes[3].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Heatmap normalizzata\n(scala globale, soglia = 0.5)")
    cbar = fig.colorbar(im_norm, ax=axes[3], fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels([
        f"{norm['v_min']:.3g}",
        f"soglia {norm['threshold']:.3g}",
        f"{norm['v_max']:.3g}",
    ])

    axes[4].imshow(make_overlay(img_rgb, amap_norm))
    if has_gt:
        axes[4].contour(gt_mask, levels=[0.5], colors="white", linewidths=1)
    axes[4].set_title("Overlay" + (" (contorno bianco = GT)" if has_gt else ""))

    for ax in axes:
        ax.axis("off")


def save_full_image_heatmap(img_rgb, gt_mask, amap, norm, title, output_dir, img_name,
                            display_scale=0.25, zoom_size=TILE_SIZE):
    """
    Salva:
      - overlay_<nome>.jpg: overlay a piena risoluzione della heatmap normalizzata
      - plot_<nome>.png: riga 1 = immagine intera (ridotta), riga 2 = zoom
        a piena risoluzione attorno al massimo della mappa
    """
    amap_norm = normalize_anomaly_map(amap, norm)

    overlay = make_overlay(img_rgb, amap_norm)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{Path(img_name).stem}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    H, W = amap.shape
    small = (max(1, int(W * display_scale)), max(1, int(H * display_scale)))
    img_s = cv2.resize(img_rgb, small, interpolation=cv2.INTER_AREA)
    amap_s = cv2.resize(amap, small, interpolation=cv2.INTER_AREA)
    # INTER_AREA + soglia > 0 per non perdere le varroe piccole nella versione ridotta
    gt_s = (cv2.resize(gt_mask.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    amap_norm_s = normalize_anomaly_map(amap_s, norm)

    cy, cx = np.unravel_index(np.argmax(amap), amap.shape)
    half = zoom_size // 2
    y0 = int(np.clip(cy - half, 0, max(0, H - zoom_size)))
    x0 = int(np.clip(cx - half, 0, max(0, W - zoom_size)))
    zoom = (slice(y0, y0 + zoom_size), slice(x0, x0 + zoom_size))

    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    _plot_row(fig, axes[0], img_s, gt_s, amap_s, amap_norm_s, norm, "immagine intera")
    axes[0][0].add_patch(plt.Rectangle(
        (x0 * display_scale, y0 * display_scale), zoom_size * display_scale, zoom_size * display_scale,
        fill=False, edgecolor="yellow", linewidth=1.5,
    ))
    _plot_row(fig, axes[1], img_rgb[zoom], gt_mask[zoom], amap[zoom], amap_norm[zoom], norm,
              f"zoom sul massimo ({cx}, {cy})")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"plot_{Path(img_name).stem}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(scores, labels, image_threshold, save_path):
    """Anomaly score per immagini pulite e con varroa."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 30) if scores.max() > scores.min() else 10
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Pulite", color="tab:blue")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Varroa", color="tab:red")
    ax.axvline(image_threshold, color="black", linestyle="--", label=f"Soglia F1 = {image_threshold:.3g}")
    ax.set_xlabel(f"Anomaly score (media top-{TOP_K} pixel)")
    ax.set_ylabel("Numero di immagini")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Test + Metriche + Visualizzazione
# ──────────────────────────────────────────────────────────────────────
def evaluate_and_visualize(model, device, output_dir, window_size=512):
    """
    Passo 1: sliding window su immagini pulite e con varroa sintetica,
             calcolo di score e pixel (sottocampionati) e salvataggio delle
             mappe su disco.
    Passo 2: metriche, normalizzazione globale e salvataggio delle heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    maps_dir = os.path.join(output_dir, "anomaly_maps")
    os.makedirs(maps_dir, exist_ok=True)

    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)
    print(f"\nImmagini pulite (label=0): {len(clean_paths)}")
    print(f"Immagini varroa (label=1): {len(varroa_paths)}")

    test_items = [(p, 0) for p in clean_paths] + [(p, 1) for p in varroa_paths]

    records = []
    all_image_labels, all_image_scores = [], []
    all_pixel_gt, all_pixel_pred, normal_pixel_pred = [], [], []

    # --- Passo 1: inferenza ---
    for img_path, label in test_items:
        img_name = os.path.basename(img_path)
        print(f"  [{'varroa' if label else 'pulita'}] {img_name}...")

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

    # --- Calcolo metriche ---
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.IMAGE),
        F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.PIXEL),
    ]

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — RD4AD Full Image")
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
            print(f"  {metric.name:>20s}: ERRORE — {e}")

    print(f"{'='*60}\n")

    # --- Passo 2: normalizzazione globale + heatmap ---
    norm = compute_heatmap_normalization(all_pixel_pred, all_pixel_gt, normal_pixel_pred)
    if 0 < all_image_labels.sum() < len(all_image_labels):
        image_threshold = f1_optimal_threshold(all_image_labels, all_image_scores)
    else:
        image_threshold = float(np.median(all_image_scores))
    print(f"Normalizzazione heatmap: v_min={norm['v_min']:.4f} | "
          f"soglia={norm['threshold']:.4f} | v_max={norm['v_max']:.4f}")
    print(f"Soglia image-level (F1 ottimale): {image_threshold:.4f}")

    save_score_distribution(all_image_scores, all_image_labels, image_threshold,
                            os.path.join(output_dir, "score_distribution.png"))

    print("\n--- Salvataggio heatmap normalizzate ---")
    for rec in records:
        img_name = os.path.basename(rec["img_path"])
        print(f"  {img_name}...")
        img_rgb = cv2.cvtColor(cv2.imread(rec["img_path"]), cv2.COLOR_BGR2RGB)
        amap = np.load(rec["map_path"]).astype(np.float32)
        gt_mask = load_gt_mask(rec["img_path"], rec["label"], amap.shape)

        status = "Varroa" if rec["label"] == 1 else "Pulita"
        pred = "Varroa" if rec["score"] >= image_threshold else "Pulita"
        title = (f"{img_name} — GT: {status} | Pred: {pred} | "
                 f"score {rec['score']:.4f} (soglia {image_threshold:.4f})")
        save_full_image_heatmap(img_rgb, gt_mask, amap, norm, title, output_dir, img_name,
                                zoom_size=window_size)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "metrics": results,
            "heatmap_normalization": norm,
            "image_threshold": image_threshold,
            "image_scores": {os.path.basename(r["img_path"]): r["score"] for r in records},
        }, f, indent=2)

    print(f"\nHeatmap salvate in {output_dir} (mappe grezze in {maps_dir})")


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

    # --- Modello ---
    model = RD4AD(backbone_name=BACKBONE, input_size=(TILE_SIZE, TILE_SIZE)).to(device)
    # l'encoder (teacher) è congelato: l'optimizer aggiorna solo bottleneck e decoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    # --- Dataset di training con crop casuali ---
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

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training RD4AD full-image | {EPOCHS} epoche | {len(train_dataloader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_dataloader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Modello RD4AD salvato in {CHECKPOINT}")

    # --- Test con sliding window + metriche + heatmap ---
    print(f"\nTest con sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
    evaluate_and_visualize(
        model, device,
        output_dir=OUTPUT_DIR,
        window_size=TILE_SIZE,
    )


if __name__ == "__main__":
    main()
