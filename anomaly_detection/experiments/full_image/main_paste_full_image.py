"""
PaSTe - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con blending delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).

Ottimizzazione per varroa:
  - wide_resnet50_2 backbone con layer poco profondi (layer1, layer2, layer3)
    per catturare dettagli tessiturali fini come il corpo della varroa.
  - PaSTe bootstrap dal layer 0 (conv1 del teacher): lo student condivide
    solo la prima convoluzione, così layer1 resta un layer di anomaly
    detection e non si perde risoluzione spaziale.
  - output_size uguale a TILE_SIZE per preservare la risoluzione
    della anomaly map a livello di pixel.

Correzioni rispetto alla versione precedente:
  - ad_layers=[1, 2, 3] invece di [0, 1, 2]. In PasteBackbone l'indice 0 è
    la sola conv1 (7x7, senza BN/ReLU): è una mappa lineare, identica tra
    teacher e student come architettura, che si comporta allo stesso modo su
    immagini normali e anomale, quindi la sua mappa di errore non contiene
    informazione sulle anomalie. Poiché le mappe dei layer vengono
    moltiplicate tra loro, quel rumore modulava tutta la anomaly map;
    inoltre il layer più profondo era layer2, con poco contesto.
  - Teacher IMAGENET1K_V1, come in main_stfpm_full_image.py.
  - 100 epoche con cosine annealing e lr 0.04, come STFPM. La loss di PaSTe
    usa MSE con reduction="sum", quindi a parità di lr il passo per campione
    non dipende dal batch size: con lr 0.01 costante e 50 epoche lo student
    riceveva circa 4 volte meno aggiornamento di quello di STFPM.
  - Heatmap normalizzate globalmente (non min/max per immagine), così le
    immagini pulite non mostrano più "anomalie" rosse fittizie.
"""

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


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/images"
TEST_CLEAN_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/images"
TEST_VARROA_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/masks"

TILE_SIZE  = 512
EPOCHS     = 100
BATCH_SIZE = 16
LR         = 0.04
SEED       = 1

# Indici dei layer di PasteBackbone per wide_resnet50_2:
#   0 = conv1 (conv 7x7 grezza, senza BN/ReLU)
#   1 = bn1 + relu + maxpool + layer1
#   2 = layer2    3 = layer3    4 = layer4
# Il layer 0 non va usato per l'anomaly detection (vedi docstring).
AD_LAYERS       = [1, 2, 3]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"

TOP_K = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / "paste"


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


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossibile leggere: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def top_k_mean(amap, k=TOP_K):
    """Media dei k pixel più alti: score robusto a singoli spike di rumore."""
    flat = amap.ravel()
    if flat.size <= k:
        return float(flat.max())
    return float(np.mean(np.partition(flat, -k)[-k:]))


class FullImageCropDataset(Dataset):
    """
    Carica immagini ad alta risoluzione e restituisce crop casuali.
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
        self.images = [read_rgb(path) for path in self.image_paths]
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
    Taglia l'immagine in tile, esegue PaSTe su ciascuna,
    e ricostruisce la heatmap globale mediando le sovrapposizioni.

    PaSTe in eval mode restituisce:
      score_maps  [B, 1, output_size_h, output_size_w]
      anomaly_scores  [B]
    dove output_size è configurato nel costruttore (= TILE_SIZE nel nostro caso).
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
            # PaSTe restituisce (score_maps [B, 1, H, W], anomaly_scores [B])
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
def load_gt_mask(img_path, shape):
    mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
    if mask_path is None:
        print(f"    ⚠ Maschera non trovata per {os.path.basename(img_path)}, uso maschera vuota")
        return np.zeros(shape, dtype=np.uint8)
    gt_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    if gt_mask.shape != shape:
        gt_mask = cv2.resize(gt_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return gt_mask


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

    image_labels = []
    image_scores = []
    amaps = []
    gt_masks = []

    for split_name, paths, label in [
        ("pulite", clean_paths, 0),
        ("varroa sintetica", varroa_paths, 1),
    ]:
        print(f"\n--- Inferenza immagini {split_name} ---")
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

    # --- Normalizzazione globale e salvataggio overlay ---
    # Stessa scala per tutte le immagini: con il min/max per immagine anche
    # quelle pulite mostravano sempre un picco rosso.
    # Sottocampioniamo per non copiare in RAM decine di immagini da 24MP.
    sample = np.concatenate([a.ravel()[::97] for a in amaps])
    vmin = np.percentile(sample, 1)
    vmax = np.percentile(sample, 99.9)
    print(f"\nNormalizzazione heatmap globale: P1={vmin:.6f}, P99.9={vmax:.6f}")

    for img_path, amap in zip(clean_paths + varroa_paths, amaps):
        _save_overlay(read_rgb(img_path), amap, output_dir,
                      os.path.basename(img_path), vmin, vmax)

    # --- Calcolo metriche ---
    image_labels = np.array(image_labels)
    image_scores = np.array(image_scores)
    pixel_gt = np.concatenate([m.ravel() for m in gt_masks])
    del gt_masks
    pixel_pred = np.concatenate([a.ravel() for a in amaps])
    del amaps

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — PaSTe Full Image")
    print(f"{'='*60}")

    for metric in metrics:
        try:
            if metric.level == MetricLvl.IMAGE:
                value = metric.compute(image_labels, image_scores)
            else:
                value = metric.compute(pixel_gt, pixel_pred)
            print(f"  {metric.name:>20s}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name:>20s}: ERRORE — {e}")

    print(f"{'='*60}\n")
    print(f"Heatmap salvate in {output_dir}")


def _save_overlay(img_rgb, amap, output_dir, img_name, vmin, vmax):
    """Salva l'overlay heatmap sull'immagine originale."""
    amap_norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Modello PaSTe ---
    # Teacher e student condividono conv1 (bootstrap 0); lo student parte
    # dall'output di conv1 del teacher e ricalcola layer1, layer2, layer3,
    # che sono i layer di anomaly detection (varroa ~2-3mm → servono i
    # layer poco profondi). output_size = (TILE_SIZE, TILE_SIZE) per mantenere
    # la risoluzione pixel-level delle anomaly maps uguale a quella del tile.
    model = PaSTe(
        backbone_model_name="wide_resnet50_2",
        input_size=(TILE_SIZE, TILE_SIZE),
        output_size=(TILE_SIZE, TILE_SIZE),
        ad_layers=AD_LAYERS,
        weights=TEACHER_WEIGHTS,
        student_bootstrap_layer=BOOTSTRAP_LAYER,
    )
    model.to(device)

    # --- Dataset di training con crop casuali ---
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

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training PaSTe full-image | {EPOCHS} epoche | {len(train_dataloader)} batch/epoca")
    print(f"AD layers: {AD_LAYERS} | bootstrap: {BOOTSTRAP_LAYER} | teacher: {TEACHER_WEIGHTS}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(EPOCHS):
        loss = model.train_epoch(epoch, train_dataloader, training_args)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

    # --- Test con sliding window + metriche ---
    print(f"\nTest con sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
    evaluate_and_visualize(
        model, device,
        output_dir=OUTPUT_DIR,
        window_size=TILE_SIZE,
    )


if __name__ == "__main__":
    main()
