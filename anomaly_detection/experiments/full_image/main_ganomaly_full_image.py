"""
GANomaly - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con blending delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).
Visualizzazione: heatmap normalizzate globalmente con percentili
                 per evidenziare le anomalie in rosso.

Correzioni rispetto alla versione precedente:
  - Input normalizzato in [-1, 1] (mean=std=0.5). Il decoder di GANomaly
    termina con una Tanh, quindi non può ricostruire valori fuori da [-1, 1]:
    con la normalizzazione ImageNet (range ~[-2.1, 2.6]) tutti i pixel chiari
    e scuri avevano un errore di ricostruzione irriducibile, e la anomaly map
    rilevava la luminosità invece della varroa.
  - Inizializzazione DCGAN effettivamente applicata ai pesi (vedi ganomaly.py).
  - Tile 128x128 invece di 256x256: GANomaly comprime l'intero tile in un
    vettore latente di 100 valori; a 256 la ricostruzione è così sfocata che
    l'errore è alto su ogni bordo/texture delle api. A 128 la ricostruzione è
    più fedele, la varroa occupa una frazione maggiore del tile e il modello
    passa da ~188M a ~50M parametri (training GAN più stabile).
  - Score image-level anche con lo score latente nativo di GANomaly
    (||z - z'||^2, massimo sui tile), oltre al top-k della anomaly map.
  - Reinizializzazione del discriminatore quando collassa (come nel paper).
  - Stesso test set (synthetic_varroa_output) degli altri script full-image.
  - Le immagini non vengono più tenute tutte in RAM per le heatmap.
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
from torch.optim import Adam
from tqdm import tqdm

from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.models.components.ganomaly.loss import GeneratorLoss, DiscriminatorLoss
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/images"
TEST_CLEAN_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/images"
TEST_VARROA_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/masks"

TILE_SIZE        = 128
LATENT_VEC_SIZE  = 100
# 80 crop da 128 coprono per epoca la stessa area di 20 crop da 256
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

# Il decoder termina con Tanh: input e ricostruzione devono stare in [-1, 1]
GANOMALY_MEAN = [0.5, 0.5, 0.5]
GANOMALY_STD  = [0.5, 0.5, 0.5]

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / "ganomaly"
CHECKPOINT = REPO_ROOT / "checkpoints" / "full_image" / "ganomaly.pth"


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
    img_name = Path(img_path).stem  # es. "DSC_5286_syn"
    base_name = re.sub(r"_syn$", "", img_name)  # es. "DSC_5286"
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
    Pre-carica tutte le immagini in memoria per evitare collo di bottiglia I/O.
    """

    def __init__(self, root_dir, crop_size=128, crops_per_image=80, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Trovate {len(self.image_paths)} immagini in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crop/immagine → {len(self)} campioni/epoca")

        # Pre-carica tutte le immagini in RAM
        print("[Dataset] Pre-caricamento in RAM delle immagini originali...")
        self.images = [read_rgb(path) for path in self.image_paths]
        print("[Dataset] Pre-caricamento completato!")

    def __len__(self):
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        img = self.images[img_idx]

        # Crop casuale
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
    Taglia l'immagine in tile, esegue GANomaly su ciascuna,
    e ricostruisce la heatmap globale con Hanning window blending.

    Restituisce l'immagine RGB, la anomaly map (errore di ricostruzione)
    e gli score latenti di GANomaly di tutti i tile.
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

    # Coordinate dei tile, assicurando copertura completa dei bordi
    y_coords = list(range(0, H - window_size + 1, stride))
    if not y_coords or y_coords[-1] + window_size < H:
        y_coords.append(max(0, H - window_size))

    x_coords = list(range(0, W - window_size + 1, stride))
    if not x_coords or x_coords[-1] + window_size < W:
        x_coords.append(max(0, W - window_size))

    # Hanning window per transizioni morbide tra tile sovrapposti
    hanning_1d = np.hanning(window_size + 2)[1:-1]
    hanning_2d = np.outer(hanning_1d, hanning_1d).astype(np.float32)

    batch_tiles = []
    batch_coords = []
    model.eval()

    def process_batch(tiles, coords):
        batch_tensor = torch.stack(tiles).to(device)
        with torch.no_grad():
            # GANomaly restituisce (anomaly_maps [B, 1, H, W], scores [B])
            batch_amaps, batch_scores = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()
            tile_scores.extend(batch_scores.cpu().numpy().tolist())

        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])  # (H_pad, W_pad)
            amap = amap[:window_size, :window_size]  # crop eventuale padding
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

    # Gaussian smoothing per ridurre il rumore pixel-level
    full_anomaly_map = cv2.GaussianBlur(full_anomaly_map, (21, 21), 0)

    return img, full_anomaly_map, np.asarray(tile_scores, dtype=np.float32)


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


def evaluate_and_visualize(model, device, output_dir):
    """
    Esegue il test su immagini pulite e con varroa sintetica,
    calcola le metriche a livello immagine e pixel, e salva le heatmap overlay.
    """
    os.makedirs(output_dir, exist_ok=True)

    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)

    print(f"\nImmagini pulite (label=0): {len(clean_paths)}")
    print(f"Immagini varroa (label=1): {len(varroa_paths)}")

    image_labels = []
    map_scores = []      # top-k della anomaly map di ricostruzione
    latent_scores = []   # score latente GANomaly, massimo sui tile
    amaps = []
    gt_masks = []

    for split_name, paths, label in [
        ("pulite", clean_paths, 0),
        ("varroa sintetica", varroa_paths, 1),
    ]:
        print(f"\n--- Inferenza immagini {split_name} ---")
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

    # --- Normalizzazione globale con percentili e salvataggio overlay ---
    # Percentili (non min/max) calcolati su tutte le mappe, così le immagini
    # pulite restano "fredde" e il rosso indica una risposta fuori scala.
    # Sottocampioniamo per non copiare in RAM decine di immagini da 24MP.
    sample = np.concatenate([a.ravel()[::97] for a in amaps])
    p_low = np.percentile(sample, 2)
    p_high = np.percentile(sample, 99.5)
    print(f"\nGlobal anomaly map range: [{sample.min():.6f}, {sample.max():.6f}]")
    print(f"Percentile normalization: P2={p_low:.6f}, P99.5={p_high:.6f}")

    for img_path, amap in zip(clean_paths + varroa_paths, amaps):
        _save_overlay(read_rgb(img_path), amap, output_dir,
                      os.path.basename(img_path), p_low, p_high)

    # --- Calcolo metriche ---
    image_labels = np.array(image_labels)
    pixel_gt = np.concatenate([m.ravel() for m in gt_masks])
    del gt_masks
    pixel_pred = np.concatenate([a.ravel() for a in amaps])
    del amaps

    image_level = [RocAuc(MetricLvl.IMAGE), AvgPrec(MetricLvl.IMAGE), F1(MetricLvl.IMAGE)]
    pixel_level = [RocAuc(MetricLvl.PIXEL), AvgPrec(MetricLvl.PIXEL), F1(MetricLvl.PIXEL)]

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — GANomaly Full Image")
    print(f"{'='*60}")

    for score_name, scores in [
        ("score immagine = top-k anomaly map", np.array(map_scores)),
        ("score immagine = latente GANomaly (max tile)", np.array(latent_scores)),
    ]:
        print(f"  [{score_name}]")
        for metric in image_level:
            _print_metric(metric, image_labels, scores)

    print("  [pixel]")
    for metric in pixel_level:
        _print_metric(metric, pixel_gt, pixel_pred)

    print(f"{'='*60}\n")
    print(f"Heatmap salvate in {output_dir}")


def _print_metric(metric, gt, pred):
    try:
        print(f"  {metric.name:>20s}: {metric.compute(gt, pred):.4f}")
    except Exception as e:
        print(f"  {metric.name:>20s}: ERRORE — {e}")


def _save_overlay(img_rgb, amap, output_dir, img_name, p_low, p_high):
    """
    Salva l'overlay heatmap sull'immagine originale.
    Normalizzazione con percentili + gamma moderato per evidenziare anomalie in rosso.
    """
    # Normalizzazione con percentili globali
    amap_norm = np.clip((amap - p_low) / (p_high - p_low + 1e-8), 0, 1)

    # Gamma moderato: amplifica le differenze senza schiacciare tutto a 0
    gamma = 1.5
    amap_norm = amap_norm ** gamma

    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)

    # Plot con le 3 viste
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Originale")
    axes[0].axis("off")

    im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap (normalizzata)")
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
    model = Ganomaly(
        input_size=(TILE_SIZE, TILE_SIZE),
        num_input_channels=3,
        n_features=64,
        latent_vec_size=LATENT_VEC_SIZE,
        extra_layers=0,
        add_final_conv_layer=True,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri GANomaly: {n_params / 1e6:.1f}M")

    # --- Dataset di training con crop casuali ---
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

    # --- Optimizer e loss (training GAN custom) ---
    g_opt = Adam(model.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(model.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    g_loss_fn = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    d_loss_fn = DiscriminatorLoss()

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training GANomaly full-image | {EPOCHS} epoche | {len(train_dataloader)} batch/epoca")
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
            # GeneratorLoss usa "feature matching", quindi passiamo feat_real e feat_fake
            g_loss = g_loss_fn(latent_i, latent_o, padded, fake, feat_real, feat_fake)

            g_opt.zero_grad()
            g_loss.backward(retain_graph=True)
            g_opt.step()
            avg_g_loss += g_loss.item()

            # Discriminator update
            pred_fake_score, _ = model.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real_score, pred_fake_score)

            d_opt.zero_grad()
            d_loss.backward()  # rimosso retain_graph=True per liberare memoria
            d_opt.step()
            avg_d_loss += d_loss.item()

            # Se il discriminatore collassa lo si reinizializza (come nel paper)
            if d_loss.item() < 1e-5:
                model.discriminator.apply(Ganomaly.weights_init)

        avg_g_loss /= len(train_dataloader)
        avg_d_loss /= len(train_dataloader)
        print(f"Epoca {epoch+1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")

    # --- Salvataggio modello ---
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Modello GANomaly salvato in {CHECKPOINT}")

    # --- Test con sliding window + metriche ---
    print(f"\nTest con sliding window (tile {TILE_SIZE}x{TILE_SIZE}, stride {INFER_STRIDE})...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
