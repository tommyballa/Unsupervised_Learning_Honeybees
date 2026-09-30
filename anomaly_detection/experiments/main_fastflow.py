"""
FastFlow - Anomaly detection su singole api (single bee crops).

Addestramento: feature wide_resnet50_2 ImageNet congelate (layer1-3),
               LayerNorm addestrabili per scala e un normalizing flow 2D per
               scala, massimizzando la likelihood delle api sane (VarroaDataset,
               TRAIN). Adam lr 1e-3, wd 1e-5, cosine annealing.
Valutazione:   mappa = media sulle scale di mean(z^2) smussata (gaussiano
               sigma 4), score immagine = massimo della mappa; metriche a
               livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO).
Visualizzazione: heatmap normalizzate GLOBALMENTE e ancorate alla soglia pixel
                 F1-ottimale (moviad.utilities.evaluation.report).

Correzioni rispetto alla versione precedente:
  - LayerNorm per scala riattivate (create_fastflow(normalize=True)), come
    nel paper e in anomalib;
  - valutazione in fp32 con metriche anche a livello pixel (prima solo
    immagine, in bf16) e heatmap con normalizzazione globale;
  - loop di training diretto invece di ContinualTrainer + SingleModel.
Il checkpoint va passato a diagnose_fastflow_varroa.py con --checkpoint; i
checkpoint addestrati senza LayerNorm non sono compatibili con il modello.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

# cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2
BACKBONE      = "wide_resnet50_2"
IMG_SIZE      = (256, 256)
EPOCHS        = 300
BATCH_SIZE    = 16
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
EVAL_INTERVAL = 10           # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = "./heatmaps_fastflow_debug"
CHECKPOINT    = "./checkpoints/fastflow_single_bee.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True è sicuro con input size fisso e velocizza le convoluzioni
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


def train_fastflow_single_bee():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Modello (LayerNorm per scala + normalizing flow) ---
    model = create_fastflow(img_shape=IMG_SIZE, backbone_name=BACKBONE, device=device, normalize=True)

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

    # --- Optimizer (flow + LayerNorm), scheduler per epoca e loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    training_args = FastFlowTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      optimizer=optimizer, scheduler=scheduler)
    training_args.init_train(model)

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training FastFlow single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_loader, training_args)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello FastFlow salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="FastFlow single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_fastflow_single_bee()
