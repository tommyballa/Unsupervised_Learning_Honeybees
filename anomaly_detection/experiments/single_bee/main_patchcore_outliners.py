import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from moviad.models import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.configurations import Split


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_and_find_outliers():
    set_seed(1)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\nDevice in use: {device}")

    # 1. Configurazione dataset (impostazioni base usate per PatchCore)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    dataset_args = DatasetArguments(
        dataset_path="/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee/train_single_bee_clustered/1_nitida_intera",
        img_size=(512, 512),
        gt_mask_size=(512, 512),
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            imagenet_normalize,
        ],
    )

    # 2. Caricamento del dataset di training (tutte api normali/senza label)
    print("\nCaricamento del training dataset (VarroaDataset - Split.TRAIN)...")
    full_train_dataset = VarroaDataset(
        dataset_args,
        category="varroa",
        split=Split.TRAIN
    )

    # 3. Suddivisione 80% train, 20% val
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    print(f"Dimensione totale: {dataset_size}")
    print(f"Dimensione Train (80%): {train_size}")
    print(f"Dimensione Val (20%): {val_size}")

    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
    # Batch size 1 per l'inferenza semplificata o mappatura diretta path
    val_dataloader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=4)

    # 4. Inizializzazione modello PatchCore
    print("\nInizializzazione PatchCore (wide_resnet50_2)...")
    feature_extractor = CustomFeatureExtractor(
        "wide_resnet50_2",
        ["layer1", "layer2"],
        device,
        frozen=True
    )
    # Riduciamo un po' il memory_bank_size se necessario (Patchcore standard è 10-20k)
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=20000)
    model.to(device)

    # 5. Addestramento (Estrazione Feature + Costruzione Coreset)
    print("\nAddestramento in corso (estrazione feature e coreset)...")
    model.train()
    training_args = TrainingArgs(batch_size=8, epochs=1)
    
    # PatchCore "addestra" ignorando i veri loop di epochs ma popolando il memory bank
    model.train_epoch(0, train_dataloader, training_args)

    print("\nAddestramento completato. Inizio rilevamento outliers sul validation set...")
    
    # 6. Outlier Detection su Validation Set
    model.eval()
    results = []

    with torch.no_grad():
        # Essendo il Dataloader batch=1, possiamo accoppiarlo direttamente con gli indici del val_subset
        for i, batch in enumerate(val_dataloader):
            # Nel dataset TRAIN, il batch è direttamente il tensore dell'immagine
            image = batch.to(device)
            
            # PatchCore in eval() restituisce (anomaly_maps, pred_scores)
            anomaly_maps, anomaly_scores = model(image)
            
            score = anomaly_scores.item()
            amap = anomaly_maps.cpu().numpy()[0, 0] # Prendi mappa singola (H, W)
            
            # Recuperiamo il path originale dell'immagine
            original_idx = val_subset.indices[i]
            img_path = full_train_dataset.samples.iloc[original_idx]["image_path"]

            results.append({
                "path": img_path,
                "score": score,
                "map": amap
            })

    # 7. Ordinamento per trovare i più anomali (Outliers)
    results.sort(key=lambda x: x["score"], reverse=True)

    top_n = 20
    top_results = results[:top_n]
    
    print(f"\n--- TOP {top_n} OUTLIERS NEL VALIDATION SET ---")
    for i, res in enumerate(top_results):
        print(f"Rank #{i+1} | Score: {res['score']:>6.2f} | Immagine: {Path(res['path']).name}")

    # 8. Salvataggio Heatmap
    output_dir = Path(__file__).resolve().parents[2] / "heatmaps" / "single_bee" / "patchcore" / "outliers_train"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcolo min/max sui top risultati per normalizzazione
    global_min = min([res["map"].min() for res in top_results])
    global_max = max([res["map"].max() for res in top_results])
    
    print(f"\nSalvataggio heatmap in '{output_dir}'...")

    for i, res in enumerate(top_results):
        img_path = res["path"]
        score = res["score"]
        amap = res["map"]

        img_tensor = Image_loader(img_path)
        img_denorm = img_tensor

        amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
        amap_norm = amap_norm ** 3.0 
        
        heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_denorm, 0.4, heatmap, 0.6, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_denorm); axes[0].set_title(f"Rank #{i+1}"); axes[0].axis("off")
        
        im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Score: {score:.2f}"); axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
        plt.tight_layout()
        
        out_filename = f"outlier_rank{i+1}_score{score:.2f}_{Path(img_path).name}"
        plt.savefig(os.path.join(output_dir, out_filename), dpi=150, bbox_inches="tight")
        plt.close()
        
def Image_loader(img_path):
    from PIL import Image
    image = Image.open(img_path).convert("RGB")
    image = image.resize((512, 512))
    return np.array(image)

if __name__ == "__main__":
    train_and_find_outliers()
