"""
CFA (Coupled-hypersphere-based Feature Adaptation) - Anomaly detection su
singole api (single bee crops).

Addestramento: backbone wide_resnet50_2 ImageNet congelato (layer1 + layer2),
               memory bank inizializzata con le feature adattate delle api
               sane e ridotta con KMeans (gamma_c), patch descriptor
               addestrato con la soft-boundary loss (AdamW lr 1e-3,
               wd 5e-4, amsgrad, 30 epoche come nel paper).
Valutazione:   distanza dai centroidi più vicini, upsampling + gaussiano
               (sigma 4), score immagine = massimo della heatmap (paper);
               metriche a livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

Correzioni rispetto alla versione precedente:
  - rimosso CropBeeTransform: ritagliava l'immagine ma non la maschera GT,
    quindi mappa e maschera erano disallineate e le metriche pixel sbagliate;
  - dataset completo (VarroaDataset) invece della sola partizione
    "varroa_clean" di VarroaDomainShiftDataset (1/3 di train e test);
  - path di training corretto (prima "./data");
  - loop di training diretto (la strategia Replay valutava a ogni epoca);
  - in cfa.py: score immagine senza il margine del 35% (contava solo il 30%
    centrale del crop) e kernel gaussiano coerente con sigma 4.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.cfa.cfa import CFA, CFATrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

BACKBONE      = "wide_resnet50_2"
LAYERS        = ["layer1", "layer2"]
FEATURE_CHANNELS = 768      # layer1 (256) + layer2 (512)
# gamma_c=4: memory bank ridotta di un fattore 4 con KMeans
# gamma_d=2: canali del patch descriptor dimezzati (768 -> 384)
GAMMA_C       = 4
GAMMA_D       = 2
IMG_SIZE      = (384, 384)
EPOCHS        = 30
BATCH_SIZE    = 16
EVAL_INTERVAL = 5            # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = "./heatmaps_cfa_debug"
CHECKPOINT    = "./checkpoints/cfa_single_bee.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_cfa_single_bee():
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
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = CFA(
        feature_extractor=feature_extractor,
        backbone=BACKBONE,
        gamma_c=GAMMA_C,
        gamma_d=GAMMA_D,
        feature_maps_channels=FEATURE_CHANNELS,
    )
    model.to(device)

    # --- Dataset ---
    dataset_args = DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=IMG_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    train_dataset = VarroaDataset(dataset_args, "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args, "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # AdamW(lr 1e-3, wd 5e-4, amsgrad) su patch descriptor e raggio + soft-boundary loss
    training_args = CFATrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE)
    training_args.init_train(model)

    # --- Training (all'epoca 0 train_epoch inizializza la memory bank) ---
    print(f"\n{'='*60}")
    print(f"Training CFA single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = model.train_epoch(epoch, train_loader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello CFA salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="CFA single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_cfa_single_bee()
