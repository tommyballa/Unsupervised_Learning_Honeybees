import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from moviad.models.fastflow.fastflow import create_fastflow

class UnlabeledImageDataset(Dataset):
    """Dataset semplice che carica tutte le immagini da una cartella, senza label."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        # Supporta le estensioni di immagine più comuni
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.image_paths = sorted([
            p for p in self.folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, str(img_path)


def find_top_anomalies(model_checkpoint, data_dir, output_dir, top_n=10):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 1. Definizione delle trasformazioni
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True),
        transforms.GaussianBlur(kernel_size=3, sigma=0.5),
        imagenet_normalize,
    ])

    # 2. Inizializzazione del modello
    print("Inizializzazione del modello FastFlow...")
    model = create_fastflow(
        img_shape=(512, 512),
        backbone_name="wide_resnet50_2", 
        device=device
    )
    
    # Carica i pesi salvati dal training (se disponibili)
    if os.path.exists(model_checkpoint):
        print(f"Caricamento pesi da {model_checkpoint}...")
        # Adatta il caricamento a seconda di come sono stati salvati i pesi
        state_dict = torch.load(model_checkpoint, map_location=device, weights_only=True)
        # Se il checkpoint contiene altre informazioni oltre al state_dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # model.load_state_dict(state_dict)  # Decommenta quando il checkpoint è disponibile
    else:
        print(f"ATTENZIONE: Checkpoint {model_checkpoint} non trovato. Verrà usato il modello non addestrato (solo feature extractor pre-addestrato).")

    model.eval()

    # 3. Preparazione del DataLoader
    dataset = UnlabeledImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    if len(dataset) == 0:
        print(f"Nessuna immagine trovata in {data_dir}.")
        return

    print(f"Inizio inferenza su {len(dataset)} immagini...")
    
    results = []

    # 4. Inferenza ed estrazione anomaly scores
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            anomaly_maps, anomaly_scores = model(images)
            
            anomaly_scores = anomaly_scores.cpu().numpy()
            anomaly_maps = anomaly_maps.cpu().numpy()
            
            for i in range(len(paths)):
                results.append({
                    "path": paths[i],
                    "score": float(anomaly_scores[i]),
                    "map": anomaly_maps[i]
                })

    # 5. Ordinamento decrescente in base all'anomaly score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    top_results = results[:top_n]
    print(f"\nI {top_n} campioni più anomali:")
    for i, res in enumerate(top_results):
        print(f"{i+1}. {Path(res['path']).name} - Score: {res['score']:.4f}")

    # 6. Salvataggio delle heatmap dei top N
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcolo min/max globale solo sui top_n per la normalizzazione visiva, 
    # oppure su tutto il dataset se preferisci (qui usiamo i top_n per avere buon contrasto)
    global_min = min([res["map"].min() for res in top_results])
    global_max = max([res["map"].max() for res in top_results])

    print(f"\nSalvataggio heatmap in {output_dir}...")
    for i, res in enumerate(top_results):
        img_path = res["path"]
        score = res["score"]
        amap = res["map"]

        # Carica l'immagine originale per l'overlay
        img_orig = Image.open(img_path).convert("RGB")
        img_orig = img_orig.resize((512, 512))
        img_orig = np.array(img_orig)

        amap_norm = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0, 1)
        amap_norm = amap_norm ** 3.0 # gamma correction
        
        heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_orig, 0.4, heatmap, 0.6, 0)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_orig); axes[0].set_title(f"Rank #{i+1}"); axes[0].axis("off")
        
        im = axes[1].imshow(amap_norm, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Score: {score:.2f}"); axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
        plt.tight_layout()
        
        out_filename = f"top_{i+1}_score_{score:.2f}_{Path(img_path).name}"
        plt.savefig(os.path.join(output_dir, out_filename), dpi=150, bbox_inches="tight")
        plt.close()

    print("Completato!")

if __name__ == "__main__":
    # Esempio di utilizzo
    model_checkpoint = "./checkpoints/task_0.pth"
    
    # Sostituisci con la cartella di cui vuoi trovare le anomalie
    data_dir = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/test_single_bee/normal" 
    output_dir = "./heatmaps_top_anomalies"
    
    find_top_anomalies(model_checkpoint, data_dir, output_dir, top_n=10)
