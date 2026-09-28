from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.single_model import SingleModel
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
from torchvision import transforms
import torch
import random
import numpy as np
import os
import cv2


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


def train_fastflow():
    SEEDS = [1]
    
    for seed in SEEDS:
        set_seed(seed)
        
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        args = {
            "dataset_path": "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees",
            "img_size": (256, 256),
            "gt_mask_size": (256, 256),
            "image_transform_list": [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                imagenet_normalize,
            ],
        }

        # cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2
        model = create_fastflow(
            img_shape=(256, 256),
            backbone_name="wide_resnet50_2", 
            device=device
        )
        
        continual_dataset = ContinualDataset(
            DatasetArguments(**args),
            VarroaDataset,
            categories=["varroa"],
        )

        os.makedirs("./checkpoints", exist_ok=True)
        
        continual_model = SingleModel(model, save_directory_path="./checkpoints")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

        training_args = FastFlowTrainArgs(
            epochs=300,
            batch_size=16,
            evaluation_epoch_interval=10,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
        trainer = ContinualTrainer(
            continual_dataset,
            continual_model,
            device,
            metrics=[
                RocAuc(MetricLvl.IMAGE),
                AvgPrec(MetricLvl.IMAGE),
                F1(MetricLvl.IMAGE),
            ],
            training_args=training_args,
        )

        trainer.train()

        # ==========================================
        # SAVE TOP N ANOMALOUS IMAGES HEATMAPS
        # ==========================================
        TOP_N = 10
        print(f"Evaluating and saving heatmaps for top {TOP_N} most anomalous images (seed {seed})...")
        output_dir = f"./heatmaps_fastflow_seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        
        test_dataset = continual_dataset.get_task_data_evaluation(0)
        eval_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        continual_model.eval()
        all_maps = []
        all_scores = []
        all_paths = []
        
        for image, label, mask, path in eval_dataloader:
            with torch.no_grad():
                anom_maps, anom_scores = continual_model(image.to(device))
            
            all_maps.append(anom_maps.cpu().numpy())
            all_scores.append(anom_scores.cpu().numpy())
            all_paths.extend(path)
            
        all_maps = np.concatenate(all_maps)
        all_scores = np.concatenate(all_scores)
        
        # Min-max normalization for maps
        min_val = all_maps.min()
        max_val = all_maps.max()
        norm_maps = (all_maps - min_val) / (max_val - min_val + 1e-8)
        
        # Get top N anomalous images
        top_n_indices = np.argsort(all_scores)[-TOP_N:][::-1]
        
        for rank, idx in enumerate(top_n_indices):
            score = all_scores[idx]
            img_path = all_paths[idx]
            anom_map = norm_maps[idx].squeeze()
            
            original_img = cv2.imread(img_path)
            if original_img is None:
                continue
                
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            anom_map_resized = cv2.resize(anom_map, (original_img.shape[1], original_img.shape[0]))
            
            heatmap = cv2.applyColorMap(np.uint8(255 * anom_map_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            filename = f"top_{rank+1}_score_{score:.4f}_{os.path.basename(img_path)}"
            cv2.imwrite(os.path.join(output_dir, filename), overlay_bgr)
            
        print(f"Top {TOP_N} heatmaps saved to {output_dir}")

if __name__ == "__main__":
    train_fastflow()
