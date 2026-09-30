"""
PatchCore — Anomaly detection su singole api (dataset Ceschi).

Addestramento: feature wide_resnet50_2 ImageNet congelate (layer1-3) delle api
               sane (VarroaDataset, TRAIN), 10% delle patch di ogni immagine,
               memory bank ridotta a 20000 patch con coreset greedy.
Valutazione:   mappa = distanza dal vicino più prossimo nella memory bank
               (blur gaussiano), score immagine = media ripesata delle 5 patch
               peggiori; metriche a livello immagine e pixel (AUROC, AvgPrec,
               F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report), come per gli
                 altri modelli (prima: min/max globale + gamma 3).

Il training usa PatchCoreCL (un solo task) come nella versione precedente, ma
senza ContinualTrainer: la valutazione avviene una sola volta, insieme alle
heatmap.
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
from moviad.models import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.patchcore_cl import PatchCoreCL
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
LAYERS       = ["layer1", "layer2", "layer3"]
MEMORY_BANK_SIZE = 20000
IMG_SIZE     = (512, 512)
BATCH_SIZE   = 32
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # metà sane, metà con varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee" / "patchcore"
CHECKPOINT   = REPO_ROOT / "checkpoints" / "single_bee" / "patchcore.pth"


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_patchcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Modello ---
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=MEMORY_BANK_SIZE)
    patchcore = PatchCoreCL(model)

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

    # --- Training: memory bank (un solo task) ---
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    patchcore.train()
    patchcore.start_task(0, train_dataset, training_args)
    patchcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save({"memory_bank": model.memory_bank[0].cpu()}, CHECKPOINT)
    print(f"\nMemory bank PatchCore salvata in {CHECKPOINT}")

    # --- Valutazione + heatmap normalizzate ---
    evaluate_and_report(
        patchcore, test_dataset, test_loader, device, OUTPUT_DIR,
        title="PatchCore single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_patchcore_single_bee()
