"""
PaSTe (Partially Shared Teacher-student) - Anomaly detection su singole api
(single bee crops).

Addestramento: teacher wide_resnet50_2 ImageNet congelato; lo student (pesi
               casuali) parte dall'output di conv1 del teacher (bootstrap
               layer 0) e impara a riprodurre le feature di layer1-3 sulle api
               sane (VarroaDataset, TRAIN). SGD lr 0.04 con cosine annealing.
Valutazione:   mappa = prodotto delle distanze (feature normalizzate) sui
               layer AD, score immagine = massimo della mappa; metriche a
               livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

Correzioni rispetto alla versione precedente:
  - normalizzazione ImageNet dell'input (prima image_transform_list=None);
  - path di training corretto (prima "./data");
  - ad_layers [1, 2, 3] invece di [2, 3, 4]: layer4 a 224 px è una griglia
    7x7, troppo grossolana per la varroa; bootstrap da conv1 così layer1
    resta un layer AD (come in main_paste_full_image.py);
  - lr 0.04 con cosine annealing (la loss usa MSE con reduction="sum");
  - un solo seed, come gli altri script sulla singola ape.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.paste.paste import PaSTe, PasteTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

BACKBONE        = "wide_resnet50_2"
AD_LAYERS       = [1, 2, 3]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"
IMG_SIZE        = (256, 256)
EPOCHS          = 20
BATCH_SIZE      = 32
LR              = 0.04
EVAL_INTERVAL   = 5          # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS     = 8
SEED            = 1

NUM_HEATMAPS    = 20         # metà sane, metà con varroa
OUTPUT_DIR      = "./heatmaps_paste_debug"
CHECKPOINT      = "./checkpoints/paste_single_bee.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def train_paste_single_bee():
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
    model = PaSTe(
        backbone_model_name=BACKBONE,
        input_size=IMG_SIZE,
        output_size=IMG_SIZE,
        ad_layers=AD_LAYERS,
        weights=TEACHER_WEIGHTS,
        student_bootstrap_layer=BOOTSTRAP_LAYER,
    ).to(device)

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

    # SGD (momentum 0.9, weight decay 1e-4) sui soli parametri dello student
    training_args = PasteTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training_args.optimizer, T_max=EPOCHS)

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training PaSTe single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = model.train_epoch(epoch, train_loader, training_args)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello PaSTe salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="PaSTe single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_paste_single_bee()
