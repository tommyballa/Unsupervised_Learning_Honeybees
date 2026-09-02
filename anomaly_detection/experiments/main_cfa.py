import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_domain_shift_dataset import VarroaDomainShiftDataset
from moviad.models.cfa.cfa import CFA, CFATrainArgs
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.strategies.cfa_continual import CFAContinual
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.metrics import (
    AvgPrec,
    F1,
    MetricLvl,
    ProAuc,
    RocAuc,
)

class CropBeeTransform:
    def __init__(self, padding=15):
        self.padding = padding

    def __call__(self, img):
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(img_cv.shape[1], x + w + self.padding)
        y2 = min(img_cv.shape[0], y + h + self.padding)
        
        cropped_img = img_cv[y1:y2, x1:x2]
        
        # Convert back to PIL Image
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)


def save_anomaly_heatmaps(model, dataloader, device, output_dir="./heatmaps_debug", num_samples=15):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print("\nCalculating global min/max across the dataset...")
    global_min = float('inf')
    global_max = float('-inf')
    
    # pass 1: find dataset-wide min and max limits for normalization
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
            else:
                images = batch[0].to(device)
                
            anomaly_maps, _ = model(images)
            batch_min = anomaly_maps.min().item()
            batch_max = anomaly_maps.max().item()
            
            if batch_min < global_min:
                global_min = batch_min
            if batch_max > global_max:
                global_max = batch_max

    print(f"Global Min: {global_min:.4f} | Global Max: {global_max:.4f}")
    print(f"📸 Generating heatmaps... saving to '{output_dir}'")
    
    samples_saved = 0
    
    # pass 2: generate images using global thresholds
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
                
                # apply global normalization
                amap_norm = (amap - global_min) / (global_max - global_min + 1e-8)
                amap_norm = np.clip(amap_norm, 0, 1)
                amap_uint8 = (amap_norm * 255).astype(np.uint8)
                
                heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                overlay = cv2.addWeighted(img_denorm, 0.6, heatmap, 0.4, 0)
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                status = "Varroa" if labels[i] == 1 else "Sana"
                
                axes[0].imshow(img_denorm)
                axes[0].set_title(f"Originale ({status})")
                axes[0].axis("off")
                
                # lock colormap to global limits
                im = axes[1].imshow(amap, cmap="jet", vmin=global_min, vmax=global_max)
                axes[1].set_title(f"Heatmap (Score: {anomaly_scores[i]:.2f})")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay")
                axes[2].axis("off")
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, f"sample_{samples_saved}_{status}_score_{anomaly_scores[i]:.2f}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
                
                samples_saved += 1

    print(f"✅ Saved {samples_saved} debug images.")


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_cfa_wideresnet():
    SEEDS = [1]

    for seed in SEEDS:
        set_seed(seed)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\n Device in use: {device}")
        if torch.cuda.is_available():
            print(f" GPU: {torch.cuda.get_device_name(0)}")

        feature_extractor = CustomFeatureExtractor(
            "wide_resnet50_2", ["layer1", "layer2"], device, frozen=True
        )
        # L'uso di gamma_c e gamma_d riduce notevolmente il costo computazionale
        # gamma_c=8 raggruppa i centroidi della memory bank di un fattore 8 (via KMeans)
        # gamma_d=2 dimezza la dimensione dei canali (da 768 a 384 in questo caso)
        # layer1 (256) + layer2 (512) = 768 channels
        model = CFA(feature_extractor=feature_extractor, backbone="wide_resnet50_2", gamma_c=4, gamma_d=2, feature_maps_channels=768)
        model.to(device)

        from torchvision import transforms
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        dataset_args = DatasetArguments(
            dataset_path="./data",
            img_size=(384, 384),
            gt_mask_size=(384, 384),
            image_transform_list=[
                CropBeeTransform(padding=15),
                transforms.ToTensor(),
                transforms.Resize((384, 384), antialias=True),
                imagenet_normalize,
            ],
        )

        print("\n Building the 3 domain shift tasks...")
        continual_dataset = ContinualDataset(
            dataset_args,
            VarroaDomainShiftDataset,
            categories=["varroa_clean"],
        )

        continual_model = CFAContinual(model)

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
                
            ],
            training_args=CFATrainArgs(
                batch_size=16, 
                epochs=30,
            ),
            logger=None,
        )

        trainer.train()

        with torch.inference_mode():
            from torch.utils.data import DataLoader
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
    train_cfa_wideresnet()
