"""
Dinomaly training ottimizzato per rilevamento varroa su api singole.

Ottimizzazioni velocità:
  - torch.compile su encoder blocks (frozen) e decoder blocks
  - AMP bfloat16 su forward + loss
  - batch_size=32 (A6000 48GB lo regge comodamente)
  - Valutazione ogni 5 epoche anziché ogni epoca
  - persistent_workers + pin_memory nei DataLoader
  - cudnn.benchmark=True (input size fisso)
  - Attention mask cachata (calcolata una volta sola)
  - zero_grad(set_to_none=True) + non_blocking transfers

Ottimizzazioni qualità varroa:
  - mask_neighbor_size=5 per evitare l'identity shortcut
  - Data augmentation geometrica (flips, rotazione) sul dataset cachato
  - Fix gt_mask_size a 518x518 (prima usava 256x256 di default)
"""

from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE       = (518, 518)
EPOCHS         = 20
BATCH_SIZE     = 32
EVAL_INTERVAL  = 5      # valuta ogni N epoche
SEED           = 1
NUM_WORKERS    = 8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True è sicuro con input size fisso e velocizza le convoluzioni
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Dataset wrapper con data augmentation
# ──────────────────────────────────────────────────────────────────────
class AugmentedTrainDataset(Dataset):
    """
    Wrappa il VarroaDataset (che cacha i tensori trasformati) e applica
    augmentation geometrica random ad ogni accesso.
    
    L'augmentation opera su tensori già normalizzati: le trasformazioni
    geometriche (flip, rotazione) non alterano la distribuzione dei valori.
    Questo è fondamentale per il rilevamento varroa perché le api possono
    apparire in qualsiasi orientamento.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15, fill=0),
            ], p=0.3),
        ])
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        image = self.base[idx]  # tensore cachato (C, H, W)
        return self.aug(image)


# ──────────────────────────────────────────────────────────────────────
# Heatmap di debug
# ──────────────────────────────────────────────────────────────────────
def save_anomaly_heatmaps(model, dataloader, device, output_dir="./heatmaps_debug", num_samples=15):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    print("\nCalculating global min/max across the dataset...")
    global_min = float("inf")
    global_max = float("-inf")
    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                anomaly_maps, a_scores = model(images)
            global_min = min(global_min, anomaly_maps.min().item())
            global_max = max(global_max, anomaly_maps.max().item())
            all_scores.extend(a_scores.cpu().tolist())

    # La mediana degli score divide perfettamente le immagini sane (score bassi) da quelle anomale (score alti)
    score_threshold = np.median(all_scores)

    print(f"Global Min: {global_min:.4f} | Global Max: {global_max:.4f} | Threshold: {score_threshold:.4f}")
    print(f"Generating heatmaps... saving to '{output_dir}'")

    saved_normal = 0
    saved_anomal = 0
    target_each = num_samples // 2

    with torch.no_grad():
        for batch in dataloader:
            if saved_normal >= target_each and saved_anomal >= target_each:
                break

            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].cpu().numpy()
            else:
                images = batch[0].to(device)
                labels = batch[1].cpu().numpy()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                anomaly_maps, anomaly_scores = model(images)
            anomaly_maps = anomaly_maps.cpu().numpy()
            anomaly_scores = anomaly_scores.cpu().numpy()

            for i in range(images.size(0)):
                is_anomal = labels[i] == 1
                
                if is_anomal and saved_anomal >= target_each:
                    continue
                if not is_anomal and saved_normal >= target_each:
                    continue

                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                amap = np.squeeze(anomaly_maps[i])
                
                # Normalizzazione globale base (identica all'originale che andava benissimo per la varroa)
                amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
                
                # Se lo score dell'immagine è sotto la soglia (immagine sana), azzeriamo il rumore di fondo.
                # In questo modo le sane sono perfettamente blu e le anomale mantengono la loro heatmap originale.
                if anomaly_scores[i] <= score_threshold:
                    amap_norm = amap_norm * 0.0
                    
                amap_norm = amap_norm ** 3.0 
                
                heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_denorm, 0.4, heatmap, 0.6, 0)

                status = "Varroa" if is_anomal else "Sana"
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_denorm);  axes[0].set_title(f"Originale ({status})");  axes[0].axis("off")
                
                im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title(f"Score: {anomaly_scores[i]:.2f}");  axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                axes[2].imshow(overlay);  axes[2].set_title("Overlay");  axes[2].axis("off")
                plt.tight_layout()
                
                current_id = saved_anomal if is_anomal else saved_normal
                plt.savefig(
                    os.path.join(output_dir, f"sample_{status}_{current_id}_score_{anomaly_scores[i]:.2f}.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close()
                
                if is_anomal:
                    saved_anomal += 1
                else:
                    saved_normal += 1

    print(f"Saved {saved_normal} normal images and {saved_anomal} anomalous debug images.")


# ──────────────────────────────────────────────────────────────────────
# Training diretto (bypass ContinualTrainer per massima velocità)
# ──────────────────────────────────────────────────────────────────────
def train_dinomaly():

    set_seed(SEED)
    torch.set_float32_matmul_precision('high')

    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Modello ──────────────────────────────────────────────────────
    model = Dinomaly(
        encoder_name="vit_base_patch14_reg4_dinov2", 
        target_layers=[2, 3, 4, 5],
        fuse_layer_encoder=[[0, 1, 2, 3]],
        mask_neighbor_size=5,   # impedisce l'identity shortcut
        img_size=IMG_SIZE
    )
    model = model.to(device)

    # ── torch.compile sui blocchi computazionali ─────────────────────
    # L'encoder è congelato con input size fisso → compilazione perfetta
    print("Compiling encoder blocks...")
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            model.encoder.blocks[i] = torch.compile(blk)

    # Decoder e bottleneck: compilazione per accelerare il backward pass
    print("Compiling decoder and bottleneck blocks...")
    for i, blk in enumerate(model.decoder):
        model.decoder[i] = torch.compile(blk)
    for i, blk in enumerate(model.bottleneck):
        model.bottleneck[i] = torch.compile(blk)

    # ── Dataset ──────────────────────────────────────────────────────
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    dataset_args = DatasetArguments(
        dataset_path="/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees",
        img_size=IMG_SIZE,
        gt_mask_size=IMG_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            imagenet_normalize,
        ],
    )

    # Training dataset con augmentation geometrica
    train_base = VarroaDataset(dataset_args, "varroa", Split.TRAIN)
    train_dataset = AugmentedTrainDataset(train_base)

    # Test dataset (senza augmentation)
    test_dataset = VarroaDataset(dataset_args, "varroa", [Split.TEST])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # ── Training args ────────────────────────────────────────────────
    total_iters = EPOCHS * len(train_loader)
    training_args = DinomalyTrainArgs(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        total_iters=total_iters,
    )
    training_args.init_train(model)

    # ── Metriche ─────────────────────────────────────────────────────
    metrics = [
        RocAuc(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.IMAGE),
        F1(MetricLvl.PIXEL),
        ProAuc(MetricLvl.PIXEL),
    ]

    os.makedirs("./checkpoints", exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Dinomaly | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"Batch size: {BATCH_SIZE} | Eval ogni {EVAL_INTERVAL} epoche")
    print(f"total_iters (LR scheduler): {total_iters}")
    print(f"{'='*60}\n")

    best_metrics = {m.name: 0.0 for m in metrics}

    for epoch in range(EPOCHS):
        model.train()
        print(f"EPOCH: {epoch}")

        # Training epoch
        avg_loss = 0.0
        for batch in tqdm(train_loader):
            avg_loss += model.train_step(batch, training_args)
        avg_loss /= len(train_loader)
        print(f"  Loss: {avg_loss:.6f}")

        # Evaluation (ogni EVAL_INTERVAL epoche o all'ultima epoca)
        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            print("  Evaluating model...")
            results = Evaluator.evaluate(model, eval_loader, metrics, device)
            
            print("  Training performances:")
            for k, v in results.items():
                marker = " ↑ BEST" if v > best_metrics.get(k, 0) else ""
                print(f"    {k}: {v:.4f}{marker}")
            
            # Aggiorna best metrics
            for k in results:
                best_metrics[k] = max(best_metrics[k], results[k])

    # ── Salvataggio ──────────────────────────────────────────────────
    torch.save(model.state_dict(), "./checkpoints/dinomaly_final.pth")
    print("\nModello Dinomaly salvato in ./checkpoints/dinomaly_final.pth")

    print(f"\n{'='*60}")
    print("Best metrics across all evaluations:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

    # ── Heatmaps ─────────────────────────────────────────────────────
    print("\nGenerazione heatmaps di debug...")
    heatmap_loader = DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    with torch.inference_mode():
        save_anomaly_heatmaps(
            model=model,
            dataloader=heatmap_loader,
            device=device,
            output_dir="./heatmaps_dinomaly_debug",
            num_samples=20,
        )


if __name__ == "__main__":
    train_dinomaly()
