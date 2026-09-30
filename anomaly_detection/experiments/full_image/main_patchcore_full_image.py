"""
PatchCore - Anomaly detection su immagini full-frame con molte api.

Addestramento: crop casuali dalle immagini ad alta risoluzione di training.
Inferenza: sliding window sull'immagine intera, con blending delle sovrapposizioni.
Valutazione: metriche a livello di immagine e pixel (AUROC, AvgPrec, F1).

La memory bank viene salvata in checkpoints/full_image/patchcore/
(best.pth + config.json). Heatmap e metriche su tutto il test set:
    python experiments/plot_heatmaps.py full_image patchcore
"""

import os
import re
import random
from pathlib import Path

import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Ensure these match your local moviad module structure
from moviad.models import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/images/original"
TEST_CLEAN_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/images"
TEST_VARROA_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/images"
TEST_MASKS_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output/masks"

TILE_SIZE  = 512
SEED       = 1

MODEL_NAME = "patchcore"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "full_image" / MODEL_NAME


# ──────────────────────────────────────────────────────────────────────
# Utilità
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 1):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


class SimpleImageDataset(Dataset):
    """
    Dataset class for loading training images.
    Pre-loads all images in RAM during __init__ to avoid massive I/O bottlenecks.
    """
    def __init__(self, root_dir, transform=None, label=0):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label
        self.image_paths = get_image_paths(root_dir)
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        print("[Dataset] Pre-loading original high-res images in RAM (4000x6016)...")
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
        print("[Dataset] Pre-loading complete!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.transform:
            img = self.transform(img)
        return img, self.label


# ──────────────────────────────────────────────────────────────────────
# Sliding window inference
# ──────────────────────────────────────────────────────────────────────
def sliding_window_inference(model, image_path, device,
                             window_size=512, stride=256, batch_size=8):
    """
    Slices a large image into tiles, runs PatchCore inference,
    and reconstructs the global heatmap to preserve pixel-level details.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    # Standard normalization without resizing
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Global maps to hold the reconstructed results
    full_anomaly_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Ensure we cover the entire image by snapping the last window to the bottom/right edges
    y_coords = list(range(0, H - window_size + 1, stride))
    if not y_coords or y_coords[-1] + window_size < H:
        y_coords.append(max(0, H - window_size))
        
    x_coords = list(range(0, W - window_size + 1, stride))
    if not x_coords or x_coords[-1] + window_size < W:
        x_coords.append(max(0, W - window_size))

    batch_tiles = []
    batch_coords = []
    model.eval()
    
    def process_batch(tiles, coords):
        batch_tensor = torch.stack(tiles).to(device)
        
        with torch.no_grad():
            batch_amaps, _ = model(batch_tensor)
            batch_amaps = batch_amaps.cpu().numpy()
        
        for i, (y, x) in enumerate(coords):
            amap = np.squeeze(batch_amaps[i])
            full_anomaly_map[y:y+window_size, x:x+window_size] += amap
            weight_map[y:y+window_size, x:x+window_size] += 1.0

    for y in y_coords:
        for x in x_coords:
            tile = img[y:y+window_size, x:x+window_size]
            batch_tiles.append(normalize(tile))
            batch_coords.append((y, x))
            
            if len(batch_tiles) == batch_size:
                process_batch(batch_tiles, batch_coords)
                batch_tiles, batch_coords = [], []
                
    if len(batch_tiles) > 0:
        process_batch(batch_tiles, batch_coords)
        
    full_anomaly_map = full_anomaly_map / (weight_map + 1e-8)
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
            stride=window_size // 2,
            batch_size=8,
        )

        all_image_labels.append(0)
        all_image_scores.append(float(amap.max()))

        gt_mask = np.zeros_like(amap, dtype=np.float32)
        all_pixel_gt.append(gt_mask.flatten())
        all_pixel_pred.append(amap.flatten())

        _save_overlay(img_rgb, amap, output_dir, img_name)

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

        all_image_labels.append(1)
        all_image_scores.append(float(amap.max()))

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

        _save_overlay(img_rgb, amap, output_dir, img_name)

    # --- Calcolo metriche ---
    all_image_labels = np.array(all_image_labels)
    all_image_scores = np.array(all_image_scores)
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_pred = np.concatenate(all_pixel_pred)

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — PatchCore Full Image")
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
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{img_name}"), overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# Modello, best e heatmap (usati anche da experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def build_model(device):
    # Use shallower layers (layer1, layer2) to capture fine textural details like varroa
    feature_extractor = CustomFeatureExtractor(
        "wide_resnet50_2",
        ["layer1", "layer2"],
        device,
        frozen=True
    )
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=100000)
    model.to(device)
    return model


def model_state(model):
    return {"memory_bank": model.memory_bank.detach().cpu()}


def load_model_state(model, state):
    # fp32 sul device del modello: compute_anomaly_score_old confronta le feature (fp32)
    # con le righe della memory bank indicizzate da tensori sul device
    model.memory_bank = state["memory_bank"].float().to(model.device)


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
    print(f"\nDevice in use: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # RandomCrop ensures PatchCore learns from the same scale it will be tested on
    # ToPILImage is required because cv2 reads as numpy arrays, and RandomCrop expects PIL/Tensors
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(TILE_SIZE, pad_if_needed=True),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    train_dataset = SimpleImageDataset(TRAIN_DIR, transform=train_transform, label=0)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = build_model(device)

    print("\nTraining PatchCore...")
    model.train()
    model.train_epoch(0, train_dataloader, TrainingArgs(batch_size=8, epochs=1))
    # train_epoch estrae le feature in autocast: memory bank in mezza precisione e su CPU
    load_model_state(model, model_state(model))

    save_checkpoint(CHECKPOINT_DIR, model_state(model), script_config(globals()),
                    best_epoch=None, note="memory bank costruita in un solo passaggio")
    print(f"\nMemory bank PatchCore salvata in {CHECKPOINT_DIR}")
    print(f"Heatmap e metriche su tutto il test: python experiments/plot_heatmaps.py full_image {MODEL_NAME}")


if __name__ == "__main__":
    main()
