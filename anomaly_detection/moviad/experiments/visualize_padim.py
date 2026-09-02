import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from moviad.models import Padim
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split
from torch.utils.data import DataLoader
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def denormalize(tensor):
    """Riporta il tensore PyTorch normalizzato in formato immagine RGB (H, W, C) [0, 1]."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def load_padim_model(checkpoint_path: str, device: str):
    """Carica il modello PaDiM dai pesi/statistiche salvati su disco."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato in '{checkpoint_path}'. "
            f"Esegui prima `python experiments/main_padim.py` per addestrare e salvare il modello!"
        )

    print(f"📦 Caricamento checkpoint da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = Padim(
        backbone_model_name=checkpoint.get("backbone_model_name", "mobilenet_v2"),
        layers_idxs=checkpoint.get("layers_idxs", (4, 7, 10)),
        device=device,
    ).to(device)

    # Ripristina i parametri statistici calcolati durante il training
    model.gauss_mean = checkpoint["gauss_mean"]
    model.gauss_cov = checkpoint["gauss_cov"]
    model.random_dimensions = checkpoint["random_dimensions"].to(device)

    model.eval()
    return model


def visualize_from_checkpoint(
    checkpoint_path: str = "./checkpoints/padim_varroa.pt",
    num_samples: int = 6,
    save_path: str = "./padim_visualizations.png",
    device: str = None,
):
    if device is None:
        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")
    set_seed(42)

    # 1. Carica il modello dal checkpoint
    model = load_padim_model(checkpoint_path, device=device)

    # 2. Carica il Test Dataset
    dataset_args = {
        "dataset_path": "./data",
        "img_size": (224, 224),
        "gt_mask_size": (224, 224),
        "image_transform_list": None,
    }
    args = DatasetArguments(**dataset_args)
    print("Caricamento dataset di test...")
    test_dataset = VarroaDataset(args, category="varroa", split=Split.TEST)

    # 3. Seleziona un mix bilanciato di campioni normali e anomali
    normal_indices = [
        i for i, row in test_dataset.samples.iterrows() if row.label_index == 0
    ]
    anomaly_indices = [
        i for i, row in test_dataset.samples.iterrows() if row.label_index == 1
    ]

    n_each = max(1, num_samples // 2)
    selected_indices = normal_indices[:n_each] + anomaly_indices[:n_each]
    n_rows = len(selected_indices)

    print(f"Generazione heatmap e overlay per {n_rows} immagini di test...")
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, idx in enumerate(selected_indices):
        image, label, gt_mask, path = test_dataset[idx]

        # Inferenza PaDiM (restituisce score_map e img_score)
        with torch.no_grad():
            img_input = image.unsqueeze(0).to(device)
            anomaly_map, score = model(img_input)

        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        gt_mask = gt_mask.squeeze().cpu().numpy()
        rgb_img = denormalize(image)

        # Normalizzazione mappa anomalie tra 0 e 1 per la colormap
        norm_map = (anomaly_map - anomaly_map.min()) / (
            anomaly_map.max() - anomaly_map.min() + 1e-8
        )

        label_text = "Anomalo (Varroa)" if label == 1 else "Normale"

        # 1. Immagine Originale
        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].set_title(
            f"Originale ({label_text})\nAnomaly Score: {score.item():.3f}", fontsize=11
        )
        axes[row_idx, 0].axis("off")

        # 2. Ground Truth Mask
        axes[row_idx, 1].imshow(gt_mask, cmap="gray")
        axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[row_idx, 1].axis("off")

        # 3. Anomaly Heatmap
        im_heat = axes[row_idx, 2].imshow(norm_map, cmap="jet")
        axes[row_idx, 2].set_title("Anomaly Heatmap", fontsize=11)
        axes[row_idx, 2].axis("off")
        plt.colorbar(im_heat, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # 4. Overlay
        axes[row_idx, 3].imshow(rgb_img)
        axes[row_idx, 3].imshow(norm_map, cmap="jet", alpha=0.45)
        axes[row_idx, 3].set_title("Overlay (Image + Heatmap)", fontsize=11)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Visualizzazione salvata con successo in: {save_path}")


if __name__ == "__main__":
    visualize_from_checkpoint(
        checkpoint_path="./checkpoints/padim_varroa.pt",
        num_samples=6,
        save_path="./padim_visualizations.png",
    )
