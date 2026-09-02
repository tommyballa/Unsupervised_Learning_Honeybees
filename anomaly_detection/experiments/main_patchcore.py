import os
import random
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_domain_shift_dataset import VarroaDomainShiftDataset
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

    samples_saved = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_saved >= num_samples:
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
                if samples_saved >= num_samples:
                    break

                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                amap = np.squeeze(anomaly_maps[i])
                amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
                heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_denorm, 0.6, heatmap, 0.4, 0)

                status = "Varroa" if labels[i] == 1 else "Sana"
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_denorm);  axes[0].set_title(f"Originale ({status})");  axes[0].axis("off")
                im = axes[1].imshow(amap, cmap="jet", vmin=global_min, vmax=global_max)
                axes[1].set_title(f"Heatmap (Score: {anomaly_scores[i]:.2f})");  axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                axes[2].imshow(overlay);  axes[2].set_title("Overlay");  axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"sample_{samples_saved}_{status}_score_{anomaly_scores[i]:.2f}.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close()
                samples_saved += 1

    print(f"Saved {samples_saved} debug images.")


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
        dataset_path="./data",
        img_size=(384, 384),
        gt_mask_size=(384, 384),
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize((384, 384), antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            imagenet_normalize,
        ],
    )

    # ------------------------------------------------------------------
    # Build the continual dataset (needed even for pre-training to extract
    # the healthy training split automatically)
    # ------------------------------------------------------------------
    print("\n Building the 3 domain shift tasks...")
    continual_dataset = ContinualDataset(
        dataset_args,
        VarroaDomainShiftDataset,
        categories=["varroa_clean"],
    )

    # ------------------------------------------------------------------
    # Build PatchCore model
    # ------------------------------------------------------------------
    feature_extractor = CustomFeatureExtractor(
        "wide_resnet50_2",
        ["layer1", "layer2"],
        device,
        frozen=True
    )
    model = PatchCore(feature_extractor=feature_extractor)
    continual_model = PatchCoreCL(model)

    # ------------------------------------------------------------------
    # Continual training
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
        training_args=TrainingArgs(batch_size=8, epochs=1),
        logger=None,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Evaluation + heatmaps
    # ------------------------------------------------------------------
    from torch.utils.data import DataLoader

    with torch.inference_mode():
        for task_idx in range(len(continual_dataset)):
            test_dataset = continual_dataset.get_task_data_evaluation(task_index=task_idx)
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            task_name = continual_dataset.get_task_category(task_idx)
            save_anomaly_heatmaps(
                model=continual_model,
                dataloader=test_dataloader,
                device=device,
                output_dir=f"./heatmaps_debug/task_{task_idx}_{task_name}",
                num_samples=20,
            )


if __name__ == "__main__":
    train_patchcore_wideresnet()