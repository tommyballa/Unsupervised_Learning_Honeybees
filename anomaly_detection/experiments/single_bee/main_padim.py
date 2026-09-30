"""
PaDiM - Anomaly detection su singole api (single bee crops).

Addestramento: stima di una gaussiana multivariata per posizione sulle feature
               (backbone ImageNet congelato) delle api sane (VarroaDataset, TRAIN).
               Un solo passaggio sui dati, covarianza con np.cov + 0.01 I come
               nel paper.
Valutazione:   distanza di Mahalanobis per posizione, upsampling + gaussiano
               (sigma 4), score immagine = massimo della mappa; metriche a
               livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

Correzioni rispetto alla versione precedente:
  - normalizzazione ImageNet dell'input (prima image_transform_list=None: il
    backbone pre-addestrato riceveva immagini in [0, 1] non normalizzate);
  - path di training corretto (prima "./data");
  - fitting diretto con Padim.train_epoch invece della strategia continual.
Backbone mobilenet_v2 a 224 px: con wide_resnet50_2 le feature di ~10k api
occuperebbero decine di GB di RAM (PaDiM le tiene tutte in memoria).
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
from moviad.models import Padim
from moviad.models.padim.padim import PadimTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD, evaluate_and_report


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
TRAIN_DIR    = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE      = (512, 512)

BACKBONE     = "mobilenet_v2"
LAYERS_IDXS  = (4, 7, 10)
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # metà sane, metà con varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee" / "padim"
CHECKPOINT   = REPO_ROOT / "checkpoints" / "single_bee" / "padim.pt"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_padim_single_bee():
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
    model = Padim(backbone_model_name=BACKBONE, layers_idxs=LAYERS_IDXS, device=device).to(device)

    # --- Dataset ---
    dataset_args = DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    train_dataset = VarroaDataset(dataset_args, "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args, "varroa", Split.TEST)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # --- Training: stima della gaussiana per posizione ---
    print(f"\nFitting PaDiM ({BACKBONE}, layer {LAYERS_IDXS}) su {len(train_dataset)} api sane...")
    model.train()
    model.train_epoch(0, train_loader, PadimTrainArgs(epochs=1, batch_size=BATCH_SIZE))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(
        {
            "gauss_mean": model.gauss_mean,
            "gauss_cov": model.gauss_cov,
            "random_dimensions": model.random_dimensions.cpu(),
            "backbone_model_name": model.backbone_model_name,
            "layers_idxs": LAYERS_IDXS,
        },
        CHECKPOINT,
    )
    print(f"Checkpoint PaDiM salvato in {CHECKPOINT}")

    # --- Valutazione + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="PaDiM single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_padim_single_bee()
