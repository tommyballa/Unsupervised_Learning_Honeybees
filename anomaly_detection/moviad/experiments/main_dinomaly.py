from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.single_model import SingleModel
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1
import torch
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def set_seed(seed):
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
                
                if is_anomal and saved_anomal >= target_each:
                    continue
                if not is_anomal and saved_normal >= target_each:
                    continue

                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = ((img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                amap = np.squeeze(anomaly_maps[i])
                
                # Usiamo il minimo e massimo GLOBALE, in questo modo le api sane
                # avranno valori prossimi allo zero (e quindi nessun colore rosso).
                amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
                
                # Usiamo una curva morbida per l'opacità, così i veri positivi risaltano 
                # ma il rumore di fondo nelle api sane sparisce senza essere tagliato del tutto.
                alpha_intensity = amap_norm ** 1.2
                
                heatmap = cv2.applyColorMap((alpha_intensity * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                alpha = np.expand_dims(alpha_intensity, axis=-1)
                
                # Fonde l'immagine originale e la heatmap (massimo 60% di opacità per non nascondere l'ape)
                overlay = (img_denorm * (1 - alpha * 0.6) + heatmap * (alpha * 0.6)).astype(np.uint8)

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


def train_dinomaly_cl():

    SEEDS = [1]

    for seed in SEEDS:

        set_seed(seed)

        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        print("Inizializzazione modello...")
        # NOTA: qui nel file c'era vit_base_patch16_224, ma se i pesi che vuoi caricare
        # vengono dal training di experiments/main_dinomaly.py usa vit_base_patch14_reg4_dinov2 e (518,518)
        # Per sicurezza, mettiamo le dimensioni corrette che permettono di caricare ./checkpoints/dinomaly_final.pth
        model = Dinomaly(
            encoder_name="vit_base_patch14_reg4_dinov2", 
            target_layers=[2, 3, 4, 5],
            fuse_layer_encoder=[[0, 1, 2, 3]],
            img_size=(518, 518)
        )
        model = model.to(device)

        # Carica i pesi salvati
        print("Caricamento pesi salvati in ./checkpoints/dinomaly_final.pth...")
        try:
            state_dict = torch.load("./checkpoints/dinomaly_final.pth", map_location=device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"])
            else:
                model.load_state_dict(state_dict)
            print("Pesi caricati con successo!")
        except Exception as e:
            print(f"Errore nel caricamento dei pesi: {e}")
            return

        from torchvision import transforms
        imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        args = {
            "dataset_path": "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed",
            "img_size": (518, 518),
            "gt_mask_size": (518, 518),
            "image_transform_list": [
                transforms.ToTensor(),
                transforms.Resize((518, 518), antialias=True),
                transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                imagenet_normalize,
            ],
        }

        # Singola task: tutto il dataset aggregato
        continual_dataset = ContinualDataset(
            DatasetArguments(**args),
            VarroaDataset,
            categories=["varroa"],
        )

        # Commentiamo il training:
        # continual_model = SingleModel(model)
        # training_args = DinomalyTrainArgs(epochs=5, batch_size=8, total_iters=5000)
        # trainer = ContinualTrainer(...)
        # trainer.train()

        print("Salto il training per generare direttamente le heatmap...")
        
        # Valutazione ed estrazione heatmaps per ciascun task
        with torch.inference_mode():
            for task_idx in range(len(continual_dataset)):
                test_dataset = continual_dataset.get_task_data_evaluation(task_index=task_idx)
                test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)
                task_name = continual_dataset.get_task_category(task_idx)
                
                output_dir = f"./heatmaps_dinomaly_debug/task_{task_idx}_{task_name}"
                save_anomaly_heatmaps(
                    model=model,
                    dataloader=test_dataloader,
                    device=device,
                    output_dir=output_dir,
                    num_samples=20, # numero di immagini di test da salvare
                )


if __name__ == "__main__":
    train_dinomaly_cl()
