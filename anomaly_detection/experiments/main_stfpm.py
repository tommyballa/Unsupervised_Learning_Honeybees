"""
STFPM (Student-Teacher Feature Pyramid Matching) - Anomaly detection su
singole api (single bee crops).

Addestramento: teacher resnet18 ImageNet congelato, student resnet18 con pesi
               casuali addestrato a riprodurre le feature di layer1-3 del
               teacher sulle api sane (VarroaDataset, TRAIN). Ricetta del paper:
               SGD lr 0.4, momentum 0.9, weight decay 1e-4, batch 32, 256 px.
Valutazione:   mappa = prodotto delle distanze (feature normalizzate) sui 3
               layer, score immagine = massimo della mappa; metriche a livello
               immagine e pixel (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

La versione precedente era un esperimento continual (3 task di domain shift
con Replay, path "./data"): per il continual resta main_stfpm_clad.py.
Con ~10k api, 20 epoche sono ~6k iterazioni (il paper ne fa ~1k su MVTec).
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

BACKBONE      = "resnet18"
LAYERS        = ["layer1", "layer2", "layer3"]
IMG_SIZE      = (256, 256)
EPOCHS        = 20
BATCH_SIZE    = 32
LR            = 0.4          # come nel paper STFPM
EVAL_INTERVAL = 5            # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = "./heatmaps_stfpm_debug"
CHECKPOINT    = "./checkpoints/stfpm_single_bee.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_stfpm_single_bee():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Modello: teacher pre-addestrato congelato, student con pesi casuali ---
    teacher = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    student = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=False)
    model = STFPM(teacher, student).to(device)

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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # SGD(lr, momentum 0.9, weight decay 1e-4) sui soli parametri dello student
    training_args = STFPMTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training STFPM single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca | lr {LR}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_loader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.student.model.state_dict(), CHECKPOINT)
    print(f"\nStudent STFPM salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="STFPM single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_stfpm_single_bee()
