"""
SuperSimpleNet - Anomaly detection su singole api (single bee crops).

Addestramento: immagini normali di singole api (VarroaDataset, split TRAIN).
               Le anomalie vengono simulate nello spazio delle feature
               (rumore gaussiano mascherato con Perlin noise), come nel paper.
Valutazione:   metriche a livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO)
               sul test set sintetico (api sane + api con varroa + maschere).
Visualizzazione: heatmap normalizzate GLOBALMENTE (stessa scala per tutte le
                 immagini di test) e ancorate alla soglia di decisione:
                   v_min  -> 0.0 (blu)   livello di fondo delle api sane
                   soglia -> 0.5         soglia pixel che massimizza l'F1
                   v_max  -> 1.0 (rosso) anomalie più forti del test set
                 Così un'ape sana resta blu e solo ciò che supera la soglia
                 diventa giallo/rosso (con la normalizzazione min/max per
                 immagine, invece, ogni ape avrebbe sempre un punto rosso).
"""

import json
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

BACKBONE        = "wide_resnet50_2"
LAYERS          = ["layer2", "layer3"]   # come nel paper SuperSimpleNet
IMG_SIZE        = (256, 256)
EPOCHS          = 30
BATCH_SIZE      = 32
EVAL_INTERVAL   = 5            # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS     = 8
SEED            = 1

# Iperparametri del paper / anomalib
ADAPTOR_LR      = 1e-4
SEGDEC_LR       = 2e-4
WEIGHT_DECAY    = 1e-5
LR_GAMMA        = 0.4
LR_MILESTONES   = [int(EPOCHS * 0.8), int(EPOCHS * 0.9)]   # in epoche
PERLIN_THRESHOLD = SuperSimpleNet.DEFAULT_PARAMETERS["perlin_threshold"]

NUM_HEATMAPS    = 20           # metà sane, metà con varroa
OUTPUT_DIR      = "./heatmaps_supersimplenet_debug"
CHECKPOINT      = "./checkpoints/supersimplenet_single_bee.pth"

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]


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


class AugmentedTrainDataset(Dataset):
    """Applica flip casuali alle immagini (già normalizzate) del dataset di training."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.aug(self.base[idx])


def denormalize(img_tensor):
    """Tensore normalizzato ImageNet (C, H, W) -> immagine RGB uint8 (H, W, C)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    avg_loss = 0.0
    for images in tqdm(dataloader):
        images = images.to(device)
        B, _, H, W = images.shape
        # tutte le immagini di training sono normali: maschere e label a zero,
        # le anomalie vengono generate dal modello nello spazio delle feature
        masks = torch.zeros(B, 1, H, W, device=device)
        labels = torch.zeros(B, device=device)

        anomaly_map, anomaly_score, masks, labels = model(images=images, masks=masks, labels=labels)
        loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score,
                       target_mask=masks, target_label=labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(dataloader)


# ──────────────────────────────────────────────────────────────────────
# Inferenza + metriche
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataloader, device):
    """Restituisce anomaly maps (N,1,H,W), scores (N,), labels (N,), masks GT (N,1,H,W)."""
    model.eval()
    maps, scores, labels, masks = [], [], [], []
    for images, batch_labels, batch_masks, _ in tqdm(dataloader, desc="Inferenza"):
        batch_maps, batch_scores = model(images.to(device))
        maps.append(batch_maps.float().cpu().numpy().reshape(images.shape[0], 1, *batch_maps.shape[-2:]))
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
        # nessuna maschera GT disponibile: soglia al 99.9° percentile delle api sane
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


def save_heatmap_figure(img_rgb, gt_mask, amap, norm, title, save_path):
    """
    Figura con: originale | ground truth | mappa grezza (scala della singola
    immagine) | heatmap normalizzata globalmente | overlay con contorno GT.
    """
    amap_norm = normalize_anomaly_map(amap, norm)
    has_gt = gt_mask.any()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Originale")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    im_raw = axes[2].imshow(amap, cmap="jet")
    axes[2].set_title("Mappa grezza (logit)\n(scala della singola immagine)")
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
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(scores, labels, image_threshold, save_path):
    """Istogramma degli anomaly score per api sane e con varroa."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 60) if scores.max() > scores.min() else 10
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Sane", color="tab:blue")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Varroa", color="tab:red")
    ax.axvline(image_threshold, color="black", linestyle="--", label=f"Soglia F1 = {image_threshold:.3g}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Numero di immagini")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_anomaly_heatmaps(test_dataset, maps, scores, labels, masks, output_dir, num_samples=20):
    """Calcola la normalizzazione globale e salva le heatmap di un campione bilanciato."""
    os.makedirs(output_dir, exist_ok=True)

    norm = compute_heatmap_normalization(maps, masks, maps[labels == 0])
    image_threshold = f1_optimal_threshold(labels, scores)
    print(f"\nNormalizzazione heatmap: v_min={norm['v_min']:.4f} | "
          f"soglia={norm['threshold']:.4f} | v_max={norm['v_max']:.4f}")
    print(f"Soglia image-level (F1 ottimale): {image_threshold:.4f}")

    save_score_distribution(scores, labels, image_threshold,
                            os.path.join(output_dir, "score_distribution.png"))

    rng = np.random.RandomState(SEED)
    selected = []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        selected.extend(rng.choice(idx, size=min(num_samples // 2, len(idx)), replace=False))

    print(f"Salvataggio di {len(selected)} heatmap in '{output_dir}'...")
    for i in selected:
        image, _, _, img_path = test_dataset[i]
        status = "Varroa" if labels[i] == 1 else "Sana"
        pred = "Varroa" if scores[i] >= image_threshold else "Sana"
        title = (f"{os.path.basename(img_path)} — GT: {status} | Pred: {pred} | "
                 f"score {scores[i]:.4f} (soglia {image_threshold:.4f})")
        save_heatmap_figure(
            denormalize(image), masks[i, 0], maps[i, 0], norm, title,
            os.path.join(output_dir, f"sample_{status}_{i:05d}_score_{scores[i]:.4f}.png"),
        )

    return norm, image_threshold


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def train_supersimplenet_single_bee():
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
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = SuperSimpleNet(
        feature_extractor=feature_extractor,
        perlin_threshold=PERLIN_THRESHOLD,
        stop_grad=SuperSimpleNet.DEFAULT_PARAMETERS["stop_grad"],
    ).to(device)

    # --- Dataset ---
    dataset_args = DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=IMG_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    train_dataset = AugmentedTrainDataset(VarroaDataset(dataset_args, "varroa", Split.TRAIN))
    test_dataset = VarroaDataset(dataset_args, "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Optimizer, scheduler (per epoca) e loss ---
    optimizer = AdamW([
        {"params": model.adaptor.parameters(), "lr": ADAPTOR_LR},
        {"params": model.segdec.parameters(), "lr": SEGDEC_LR, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)
    loss_fn = SSNLoss()

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training SuperSimpleNet single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            for k, v in compute_metrics(maps, scores, labels, masks, with_pro=False).items():
                print(f"    {k}: {v:.4f}")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello SuperSimpleNet salvato in {CHECKPOINT}")

    # --- Valutazione finale ---
    maps, scores, labels, masks = run_inference(model, test_loader, device)
    results = compute_metrics(maps, scores, labels, masks)

    print(f"\n{'='*60}")
    print("RISULTATI METRICHE — SuperSimpleNet single-bee")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:>20s}: {v:.4f}")
    print(f"{'='*60}")

    # --- Heatmap normalizzate ---
    norm, image_threshold = save_anomaly_heatmaps(
        test_dataset, maps, scores, labels, masks, OUTPUT_DIR, num_samples=NUM_HEATMAPS,
    )
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({"metrics": results, "heatmap_normalization": norm,
                   "image_threshold": image_threshold}, f, indent=2)
    print(f"Heatmap e risultati salvati in {OUTPUT_DIR}")


if __name__ == "__main__":
    train_supersimplenet_single_bee()
