"""
STFPM - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con blending delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).

Best: ogni EVAL_INTERVAL epoche (e all'ultima) metriche sulle prime
EVAL_IMAGES_PER_CLASS immagini pulite e con varroa del test set; la
configurazione migliore (img_roc_auc, a parità pxl_avg_prec) viene salvata in
checkpoints/full_image/stfpm/ (best.pth + config.json).
Heatmap e metriche su tutto il test set dalla best:
    python experiments/plot_heatmaps.py full_image stfpm
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import os
import re
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.full_image import eval_items, evaluate_full_images
from moviad.utilities.evaluation.report import print_metrics


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = DATA_ROOT + "/processed/train/images"
TEST_CLEAN_DIR  = DATA_ROOT + "/processed/test/images"
TEST_VARROA_DIR = DATA_ROOT + "/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = DATA_ROOT + "/processed/test/synthetic_varroa_output/masks"

TILE_SIZE  = 512
EPOCHS     = 100
BATCH_SIZE = 4
SEED       = 1

EVAL_INTERVAL = 10           # valuta ogni N epoche (e sempre all'ultima) per la best
EVAL_IMAGES_PER_CLASS = 5    # prime N immagini pulite + N con varroa del test usate per la best

MODEL_NAME = "stfpm"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "full_image" / MODEL_NAME


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


class FullImageCropDataset(Dataset):
    """
    Carica immagini ad alta resolution e restituisce crop casuali.
    Poiché il server ha molta RAM libera, pre-carica tutte le immagini in memoria
    durante l'__init__ per evitare il gravissimo collo di bottiglia I/O 
    (lettura da disco di 24MP a ogni singola iterazione).
    """

    def __init__(self, root_dir, crop_size=512, crops_per_image=10, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Trovate {len(self.image_paths)} immagini in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crop/immagine → {len(self)} campioni/epoca")
        
        print("[Dataset] Pre-caricamento in RAM delle immagini originali (4000x6016)...")
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Impossibile leggere: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
        print("[Dataset] Pre-caricamento completato!")

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
    Taglia l'immagine in tile, esegue STFPM su ciascuna,
    e ricostruisce la heatmap globale mediando le sovrapposizioni.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile leggere: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_anomaly_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

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
            # STFPM restituisce (score_maps [B, 1, H, W], anomaly_scores [B])
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
# Test + Metriche + Visualizzazione
# ──────────────────────────────────────────────────────────────────────
def evaluate_and_visualize(model, device, output_dir, window_size=512):
    """
    Esegue il test su immagini pulite e con varroa sintetica,
    calcola le metriche a livello immagine e pixel, e salva le heatmap overlay.
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

    print(f"\nImmagini pulite (label=0): {len(clean_paths)}")
    print(f"Immagini varroa (label=1): {len(varroa_paths)}")

    all_image_labels = []
    all_image_scores = []
    all_pixel_gt = []
    all_pixel_pred = []

    # --- Inferenza su immagini PULITE (label=0) ---
    print("\n--- Inferenza immagini pulite ---")
    for img_path in clean_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 4,
            batch_size=4,
        )

        amap_smoothed = cv2.GaussianBlur(amap, (21, 21), 0)

        # Usa la media dei top-100 pixel per avere uno score robusto a singoli spike di rumore
        top_k = 100
        amap_flat = amap_smoothed.flatten()
        if len(amap_flat) > top_k:
            img_score = float(np.mean(np.partition(amap_flat, -top_k)[-top_k:]))
        else:
            img_score = float(amap_smoothed.max())

        all_image_labels.append(0)
        all_image_scores.append(img_score)

        gt_mask = np.zeros_like(amap_smoothed, dtype=np.float32)
        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap_smoothed.flatten())

        _save_overlay(img_rgb, amap_smoothed, output_dir, img_name)

    # --- Inferenza su immagini VARROA (label=1) ---
    print("\n--- Inferenza immagini varroa sintetica ---")
    for img_path in varroa_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 4,
            batch_size=4,
        )

        amap_smoothed = cv2.GaussianBlur(amap, (21, 21), 0)

        top_k = 100
        amap_flat = amap_smoothed.flatten()
        if len(amap_flat) > top_k:
            img_score = float(np.mean(np.partition(amap_flat, -top_k)[-top_k:]))
        else:
            img_score = float(amap_smoothed.max())

        all_image_labels.append(1)
        all_image_scores.append(img_score)

        mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
        if mask_path:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            gt_mask = (gt_mask > 127).astype(np.float32)
            if gt_mask.shape != amap_smoothed.shape:
                gt_mask = cv2.resize(gt_mask, (amap_smoothed.shape[1], amap_smoothed.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            print(f"    ⚠ Maschera non trovata per {img_name}, uso maschera vuota")
            gt_mask = np.zeros_like(amap_smoothed, dtype=np.float32)

        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap_smoothed.flatten())

        _save_overlay(img_rgb, amap_smoothed, output_dir, img_name)

    # --- Calcolo metriche ---
    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — STFPM Full Image")
    print(f"{'='*60}")

    for metric in metrics:
        try:
            if metric.level == MetricLvl.IMAGE:
                value = metric.compute(all_image_labels, all_image_scores)
            else:
                value = metric.compute(all_pixel_gt, all_pixel_pred)
            print(f"  {metric.name:>20s}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name:>20s}: ERRORE — {e}")

    print(f"{'='*60}\n")
    print(f"Heatmap salvate in {output_dir}")


def _save_overlay(img_rgb, amap, output_dir, img_name):
    """Salva l'overlay heatmap sull'immagine originale."""
    # Rimuoviamo il rumore dai bordi usando i percentili invece del min/max assoluto
    margin = 64
    amap_center = amap[margin:-margin, margin:-margin]
    vmin = np.percentile(amap_center, 1)
    vmax = np.percentile(amap_center, 99.9)
    
    amap_norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# Modello, best e heatmap (usati anche da experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    teacher = CustomFeatureExtractor(
        "wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True
    )
    student = CustomFeatureExtractor(
        "wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False
    )
    return STFPM(teacher, student).to(device)


def model_state(model):
    # il teacher è il wide_resnet50_2 ImageNet: basta lo student
    return clean_state_dict(model.student.model.state_dict())


def load_model_state(model, state):
    model.student.model.load_state_dict(state)


def infer_image(model, device, img_path):
    """Anomaly map dell'immagine intera, come in evaluate_and_visualize."""
    _, amap = sliding_window_inference(model, img_path, device, window_size=TILE_SIZE,
                                       stride=TILE_SIZE // 4, batch_size=4)
    return cv2.GaussianBlur(amap, (21, 21), 0)


def image_score(amap, top_k=100):
    # media dei top-100 pixel, robusta a singoli spike di rumore
    flat = amap.ravel()
    if flat.size <= top_k:
        return float(flat.max())
    return float(np.mean(np.partition(flat, -top_k)[-top_k:]))


def evaluate_subset(model, device):
    """Metriche (senza plot) sulle prime EVAL_IMAGES_PER_CLASS immagini pulite e con varroa."""
    model.eval()
    items = eval_items(TEST_CLEAN_DIR, TEST_VARROA_DIR, EVAL_IMAGES_PER_CLASS)
    with torch.no_grad():
        return evaluate_full_images(lambda p: infer_image(model, device, p), image_score,
                                    items, TEST_MASKS_DIR)


def plot_heatmaps(model, device):
    """Metriche e heatmap su tutto il test set."""
    print(f"\nTest con sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
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

    # --- Dataset di training con crop casuali ---
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    training_args = STFPMTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training_args.optimizer, T_max=EPOCHS)
    tracker = BestTracker(
        CHECKPOINT_DIR, script_config(globals()),
        selection_data=(f"test set: prime {EVAL_IMAGES_PER_CLASS} immagini pulite "
                        f"+ {EVAL_IMAGES_PER_CLASS} con varroa"),
    )

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training STFPM full-image | {EPOCHS} epoche | {len(train_dataloader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_dataloader, training_args)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            results = evaluate_subset(model, device)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest {MODEL_NAME} full-image — {tracker.summary()}")
    print(f"Heatmap e metriche su tutto il test: python experiments/plot_heatmaps.py full_image {MODEL_NAME}")


if __name__ == "__main__":
    main()
