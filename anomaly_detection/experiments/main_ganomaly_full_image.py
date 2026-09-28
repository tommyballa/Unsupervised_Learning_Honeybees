"""
GANomaly - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con blending delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).
Visualizzazione: heatmap normalizzate globalmente con percentili
                 per evidenziare le anomalie in rosso.

Miglioramenti rispetto alla v1:
  - Tile 256x256 (più contesto per l'anomaly detection)
  - Gaussian smoothing sulle anomaly maps (riduce rumore pixel-level)
  - Top-k mean per image-level scoring (robusto a spike)
  - Normalizzazione con percentili (non min/max)
  - Hanning window blending (transizioni morbide tra tile)
  - Data augmentation (flip + color jitter)
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
TEST_VARROA_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output_v2/images"
TEST_MASKS_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output_v2/masks"

# 256 è potenza di 2, buon compromesso contesto/parametri per GANomaly (188M params)
TILE_SIZE  = 256
EPOCHS     = 100
BATCH_SIZE = 64
SEED       = 1
LR         = 2e-4
BETA1      = 0.5
BETA2      = 0.999

OUTPUT_DIR = "./heatmaps_ganomaly_full_image"


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


class FullImageCropDataset(Dataset):
    """
    Carica immagini ad alta risoluzione e restituisce crop casuali.
    Pre-carica tutte le immagini in memoria per evitare collo di bottiglia I/O.
    """

    def __init__(self, root_dir, crop_size=256, crops_per_image=20, transform=None):
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.transform = transform
        self.image_paths = get_image_paths(root_dir)
        print(f"[Dataset] Trovate {len(self.image_paths)} immagini in {root_dir}")
        print(f"[Dataset] {self.crops_per_image} crop/immagine → {len(self)} campioni/epoca")

        # Pre-carica tutte le immagini in RAM
        print("[Dataset] Pre-caricamento in RAM delle immagini originali...")
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
                             window_size=512, stride=256, batch_size=8):
    """
    Taglia l'immagine in tile, esegue GANomaly su ciascuna,
    e ricostruisce la heatmap globale con Hanning window blending.
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
            batch_amaps, _ = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()

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

    # Metriche da calcolare
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.IMAGE),
        F1(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.PIXEL),
    ]

    # --- Raccolta immagini di test ---
    clean_paths = get_image_paths(TEST_CLEAN_DIR)
    varroa_paths = get_image_paths(TEST_VARROA_DIR)

    print(f"\nImmagini pulite (label=0): {len(clean_paths)}")
    print(f"Immagini varroa (label=1): {len(varroa_paths)}")

    # Accumulatori
    all_image_labels = []
    all_image_scores = []
    all_pixel_gt = []
    all_pixel_pred = []
    all_amaps = []
    all_imgs = []
    all_names = []
    all_is_varroa = []

    # --- Inferenza su immagini PULITE (label=0) ---
    print("\n--- Inferenza immagini pulite ---")
    for img_path in clean_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=8,
        )

        # Image-level score: top-k mean (robusto a spike di rumore)
        top_k = 100
        amap_flat = amap.flatten()
        if len(amap_flat) > top_k:
            img_score = float(np.mean(np.partition(amap_flat, -top_k)[-top_k:]))
        else:
            img_score = float(amap.max())

        all_image_labels.append(0)
        all_image_scores.append(img_score)

        gt_mask = np.zeros_like(amap, dtype=np.float32)
        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        all_amaps.append(amap)
        all_imgs.append(img_rgb)
        all_names.append(img_name)
        all_is_varroa.append(False)

    # --- Inferenza su immagini VARROA (label=1) ---
    print("\n--- Inferenza immagini varroa sintetica ---")
    for img_path in varroa_paths:
        img_name = os.path.basename(img_path)
        print(f"  {img_name}...")

        img_rgb, amap = sliding_window_inference(
            model, img_path, device,
            window_size=window_size,
            stride=window_size // 2,
            batch_size=8,
        )

        top_k = 100
        amap_flat = amap.flatten()
        if len(amap_flat) > top_k:
            img_score = float(np.mean(np.partition(amap_flat, -top_k)[-top_k:]))
        else:
            img_score = float(amap.max())

        all_image_labels.append(1)
        all_image_scores.append(img_score)

        mask_path = find_mask_for_image(img_path, TEST_MASKS_DIR)
        if mask_path:
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            gt_mask = (gt_mask > 127).astype(np.float32)
            if gt_mask.shape != amap.shape:
                gt_mask = cv2.resize(gt_mask, (amap.shape[1], amap.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            print(f"    ⚠ Maschera non trovata per {img_name}, uso maschera vuota")
            gt_mask = np.zeros_like(amap, dtype=np.float32)

        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        all_amaps.append(amap)
        all_imgs.append(img_rgb)
        all_names.append(img_name)
        all_is_varroa.append(True)

    # --- Normalizzazione globale con percentili e salvataggio overlay ---
    # Usiamo percentili (non min/max) per escludere outlier e avere
    # una normalizzazione che evidenzi le differenze reali
    all_values = np.concatenate([a.flatten() for a in all_amaps])
    p_low = np.percentile(all_values, 2)
    p_high = np.percentile(all_values, 99.5)
    print(f"\nGlobal anomaly map range: [{all_values.min():.6f}, {all_values.max():.6f}]")
    print(f"Percentile normalization: P2={p_low:.6f}, P99.5={p_high:.6f}")

    for img_rgb, amap, img_name, is_varroa in zip(
        all_imgs, all_amaps, all_names, all_is_varroa
    ):
        _save_overlay(img_rgb, amap, output_dir, img_name, p_low, p_high)

    # --- Calcolo metriche ---
    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — GANomaly Full Image")
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
        latent_vec_size=100,
        extra_layers=0,
        add_final_conv_layer=True,
    )
    model = model.to(device)

    # --- Dataset di training con crop casuali ---
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        imagenet_normalize,
    ])

    train_dataset = FullImageCropDataset(
        TRAIN_DIR,
        crop_size=TILE_SIZE,
        crops_per_image=20,
        transform=train_transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
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

        avg_g_loss /= len(train_dataloader)
        avg_d_loss /= len(train_dataloader)
        print(f"Epoca {epoch+1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")

    # --- Salvataggio modello ---
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/ganomaly_full_image.pth")
    print("Modello GANomaly salvato in ./checkpoints/ganomaly_full_image.pth")

    # --- Test con sliding window + metriche ---
    print(f"\nTest con sliding window (tile {TILE_SIZE}x{TILE_SIZE})...")
    evaluate_and_visualize(
        model, device,
        output_dir=OUTPUT_DIR,
        window_size=TILE_SIZE,
    )


if __name__ == "__main__":
    main()
