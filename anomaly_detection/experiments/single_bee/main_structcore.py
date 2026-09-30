"""
StructCore - Anomaly detection su singole api (single bee crops).

Addestramento: come PatchCore (feature wide_resnet50_2 layer1 + layer2, 10%
               delle patch di ogni immagine, coreset greedy), più la
               calibrazione del descrittore strutturale (dispersione, media
               del top-1%, variazione totale della mappa) sulle api sane.
Valutazione:   mappa = distanza dal vicino più prossimo nella memory bank,
               score immagine = distanza di Mahalanobis diagonale del
               descrittore strutturale; metriche a livello immagine e pixel
               (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

Correzioni rispetto alla versione precedente:
  - dataset completo (VarroaDataset) invece della sola partizione
    "varroa_clean" di VarroaDomainShiftDataset (1/3 di train e test);
  - path di training corretto (prima "./data");
  - in structcore.py: il descrittore usa tutta la mappa (prima solo il 30%
    centrale del crop, margine del 35% per lato).
Il training usa StructCoreCL (un solo task) per il sottocampionamento al 10%
delle patch, necessario per non esaurire la memoria.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import StructCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.structcore_cl import StructCoreCL
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD, evaluate_and_report


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
TRAIN_DIR    = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE      = (512, 512)

BACKBONE     = "wide_resnet50_2"
LAYERS       = ["layer1", "layer2"]
IMG_SIZE     = (384, 384)
BATCH_SIZE   = 8
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # metà sane, metà con varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee" / "structcore"
CHECKPOINT   = REPO_ROOT / "checkpoints" / "single_bee" / "structcore.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_structcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Modello ---
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = StructCore(feature_extractor=feature_extractor)
    structcore = StructCoreCL(model)

    # --- Dataset ---
    dataset_args = DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    train_dataset = VarroaDataset(dataset_args, "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args, "varroa", Split.TEST)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # --- Training: memory bank + calibrazione del descrittore (un solo task) ---
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    structcore.train()
    structcore.start_task(0, train_dataset, training_args)
    structcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(
        {
            "memory_bank": model.memory_bank[0].cpu(),
            "train_mu": model.train_mu[0],
            "train_sigma": model.train_sigma[0],
        },
        CHECKPOINT,
    )
    print(f"\nMemory bank e calibrazione StructCore salvate in {CHECKPOINT}")

    # --- Valutazione + heatmap normalizzate ---
    evaluate_and_report(
        structcore, test_dataset, test_loader, device, OUTPUT_DIR,
        title="StructCore single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_structcore_single_bee()
