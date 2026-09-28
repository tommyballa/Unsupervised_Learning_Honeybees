"""
GANomaly - Anomaly detection su singole api (single bee crops).

Addestramento: usa il TrainerGanomaly con il VarroaDataset (single-bee crops).
Inferenza: forward pass diretto, con anomaly maps (errore di ricostruzione) e scores (distanza latente).
Visualizzazione: heatmap normalizzate globalmente con gamma correction
                 per evidenziare le anomalie in rosso.
"""

import os
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.models.components.ganomaly.loss import GeneratorLoss, DiscriminatorLoss
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.utilities.configurations import Split

from torch.optim import Adam
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
EPOCHS     = 100
BATCH_SIZE = 32
SEED       = 1
LR         = 2e-4
BETA1      = 0.5
BETA2      = 0.999

OUTPUT_DIR = "./heatmaps_ganomaly_debug"


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


def save_anomaly_heatmaps(model, dataloader, device, output_dir="./heatmaps_debug", num_samples=20):
    """
    Salva heatmap con normalizzazione basata su percentili e gamma correction.
    Le anomalie appaiono più rosse grazie al gamma power.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # ── Pass 1: raccogli tutti i valori per calcolare i percentili ──
    print("\nCalculating global percentiles across the dataset...")
    all_values = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["image"].to(device)
            anomaly_maps, _ = model(images)
            all_values.append(anomaly_maps.cpu().numpy().flatten())

    all_values = np.concatenate(all_values)
    # Percentili per escludere outlier e normalizzare correttamente
    p_low = np.percentile(all_values, 2)
    p_high = np.percentile(all_values, 99.5)

    print(f"Global range: [{all_values.min():.6f}, {all_values.max():.6f}]")
    print(f"Percentile normalization: P2={p_low:.6f}, P99.5={p_high:.6f}")
    print(f"Generating heatmaps... saving to '{output_dir}'")

    # ── Pass 2: genera e salva le heatmap ──
    saved_normal = 0
    saved_anomal = 0
    target_each = num_samples // 2

    with torch.no_grad():
        for batch in dataloader:
            if saved_normal >= target_each and saved_anomal >= target_each:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].cpu().numpy() if len(batch) > 1 else np.zeros(images.size(0))
            else:
                images = batch["image"].to(device)
                labels = batch["label"].cpu().numpy()

            anomaly_maps, anomaly_scores = model(images)
            anomaly_maps = anomaly_maps.cpu().numpy()
            anomaly_scores = anomaly_scores.cpu().numpy()

            for i in range(images.size(0)):
                is_anomal = labels[i] == 1

                if is_anomal and saved_anomal >= target_each:
                    continue
                if not is_anomal and saved_normal >= target_each:
                    continue

                # Denormalizza l'immagine per la visualizzazione
                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                amap = np.squeeze(anomaly_maps[i])

                # Crop alla dimensione dell'immagine originale
                # (GANomaly fa padding a potenza di 2)
                h_img, w_img = img_denorm.shape[:2]
                amap = amap[:h_img, :w_img]

                # Normalizzazione con percentili + gamma correction moderato
                # Il gamma > 1 amplifica le differenze tra zone normali e anomale
                # senza schiacciare tutto a 0 come farebbe un gamma troppo alto
                amap_norm = np.clip((amap - p_low) / (p_high - p_low + 1e-8), 0, 1)
                gamma = 1.5
                amap_norm = amap_norm ** gamma

                heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                # Resize heatmap se necessario
                if heatmap.shape[:2] != img_denorm.shape[:2]:
                    heatmap = cv2.resize(heatmap, (img_denorm.shape[1], img_denorm.shape[0]))

                overlay = cv2.addWeighted(img_denorm, 0.4, heatmap, 0.6, 0)

                status = "Varroa" if is_anomal else "Sana"
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(img_denorm)
                axes[0].set_title(f"Originale ({status})")
                axes[0].axis("off")

                im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title(f"Heatmap (Score: {anomaly_scores[i]:.4f})")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis("off")

                plt.tight_layout()
                current_id = saved_anomal if is_anomal else saved_normal
                plt.savefig(
                    os.path.join(output_dir, f"sample_{status}_{current_id}_score_{anomaly_scores[i]:.4f}.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close()

                if is_anomal:
                    saved_anomal += 1
                else:
                    saved_normal += 1

    print(f"Saved {saved_normal} normal images and {saved_anomal} anomalous debug images.")


def train_ganomaly_single_bee():
    """Training e valutazione di GANomaly su singole api."""

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
        input_size=IMG_SIZE,
        num_input_channels=3,
        n_features=64,
        latent_vec_size=100,
        extra_layers=0,
        add_final_conv_layer=True,
    )
    model = model.to(device)

    # --- Dataset ---
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    args = DatasetArguments(
        dataset_path="/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed",
        img_size=IMG_SIZE,
        gt_mask_size=IMG_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            imagenet_normalize,
        ],
    )

    continual_dataset = ContinualDataset(
        args,
        VarroaDataset,
        categories=["varroa"],
    )

    # --- Preparazione training ---
    # GANomaly ha un training loop custom con 2 optimizer (gen + disc)
    train_dataset, val_dataset, test_dataset = continual_dataset.get_task_data(0)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    g_opt = Adam(model.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(model.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    g_loss_fn = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    d_loss_fn = DiscriminatorLoss()

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training GANomaly single-bee | {EPOCHS} epoche | {len(train_dataloader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        avg_g_loss = 0.0
        avg_d_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)

            # Forward pass
            padded, fake, latent_i, latent_o = model(images)
            pred_real_score, feat_real = model.discriminator(padded)

            # Generator update
            pred_fake_score, feat_fake = model.discriminator(fake)
            g_loss = g_loss_fn(latent_i, latent_o, padded, fake, feat_real, feat_fake)

            g_opt.zero_grad()
            g_loss.backward(retain_graph=True)
            g_opt.step()
            avg_g_loss += g_loss.item()

            # Discriminator update
            pred_fake_score, _ = model.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real_score, pred_fake_score)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            avg_d_loss += d_loss.item()

        avg_g_loss /= len(train_dataloader)
        avg_d_loss /= len(train_dataloader)
        print(f"Epoca {epoch+1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")

    # --- Salvataggio modello ---
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/ganomaly_single_bee.pth")
    print("Modello GANomaly salvato in ./checkpoints/ganomaly_single_bee.pth")

    # --- Valutazione ed estrazione heatmaps ---
    print("\nGenerazione heatmaps di debug...")
    with torch.inference_mode():
        for task_idx in range(len(continual_dataset)):
            test_dataset = continual_dataset.get_task_data_evaluation(task_index=task_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            task_name = continual_dataset.get_task_category(task_idx)
            save_anomaly_heatmaps(
                model=model,
                dataloader=test_dataloader,
                device=device,
                output_dir=f"{OUTPUT_DIR}/task_{task_idx}_{task_name}",
                num_samples=20,
            )


if __name__ == "__main__":
    train_ganomaly_single_bee()
