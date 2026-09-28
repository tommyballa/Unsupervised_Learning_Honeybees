"""
PatchCore — Anomaly detection su singole api (dataset Ceschi).

Training: immagini normali da categoria_1 (10383 crop di api sane).
Test: anomaly_dataset_synthetic con api normali, anomale e maschere.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.strategies.patchcore_cl import PatchCoreCL
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import (
    AvgPrec,
    F1,
    MetricLvl,
    ProAuc,
    RocAuc,
)


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def save_anomaly_heatmaps(model, dataloader, device, output_dir="./heatmaps_debug", num_samples=15):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    print("\nCalculating global min/max across the dataset...")
    global_min = float("inf")
    global_max = float("-inf")

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
            anomaly_maps, _ = model(images)
            global_min = min(global_min, anomaly_maps.min().item())
            global_max = max(global_max, anomaly_maps.max().item())

    print(f"Global Min: {global_min:.4f} | Global Max: {global_max:.4f}")
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

            anomaly_maps, anomaly_scores = model(images)
            anomaly_maps = anomaly_maps.cpu().numpy()
            anomaly_scores = anomaly_scores.cpu().numpy()

            for i in range(images.size(0)):
                is_anomal = labels[i] == 1
                
                # Skip if we already have enough of this class
                if is_anomal and saved_anomal >= target_each:
                    continue
                if not is_anomal and saved_normal >= target_each:
                    continue

                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                amap = np.squeeze(anomaly_maps[i])
                
                # Normalizzazione GLOBALE invece che per-immagine.
                # Se usassimo min/max dell'immagine stessa, il punto "meno sano" di un'ape 
                # perfettamente sana verrebbe forzato a 1.0 (rosso). Usando il massimo globale,
                # un'ape sana con score 0.6 rispetto a un max globale di 3.0 rimarrà completamente blu!
                amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
                
                # Gamma > 1 per sopprimere i valori medi ed esaltare il vero picco (varroa)
                amap_norm = amap_norm ** 3.0 
                
                heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_denorm, 0.4, heatmap, 0.6, 0)

                status = "Varroa" if is_anomal else "Sana"
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_denorm);  axes[0].set_title(f"Originale ({status})");  axes[0].axis("off")
                
                # Mostriamo l'amap_norm scalato da 0 a 1 (poiché abbiamo già applicato il global_max e il gamma)
                im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title(f"Heatmap (Score: {anomaly_scores[i]:.2f})");  axes[1].axis("off")
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


def train_patchcore_wideresnet():
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n Device in use: {device}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    dataset_args = DatasetArguments(
        dataset_path="/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees",
        img_size=(512, 512),
        gt_mask_size=(512, 512),
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            imagenet_normalize,
        ],
    )

    # ------------------------------------------------------------------
    # Build the dataset (single task: "varroa")
    # ------------------------------------------------------------------
    print("\n Building the varroa dataset (Ceschi)...")
    continual_dataset = ContinualDataset(
        dataset_args,
        VarroaDataset,
        categories=["varroa"],
    )

    # ------------------------------------------------------------------
    # Build PatchCore model
    # ------------------------------------------------------------------
    feature_extractor = CustomFeatureExtractor(
        "wide_resnet50_2",
        ["layer1", "layer2", "layer3"],
        device,
        frozen=True
    )
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=20000)
    continual_model = PatchCoreCL(model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = ContinualTrainer(
        continual_dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        training_args=TrainingArgs(batch_size=32, epochs=1),
        logger=None,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Evaluation + heatmaps
    # ------------------------------------------------------------------
    with torch.inference_mode():
        for task_idx in range(len(continual_dataset)):
            test_dataset = continual_dataset.get_task_data_evaluation(task_index=task_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            task_name = continual_dataset.get_task_category(task_idx)
            save_anomaly_heatmaps(
                model=continual_model,
                dataloader=test_dataloader,
                device=device,
                output_dir=f"./heatmaps_debug/patchcore",
                num_samples=20,
            )


if __name__ == "__main__":
    train_patchcore_wideresnet()