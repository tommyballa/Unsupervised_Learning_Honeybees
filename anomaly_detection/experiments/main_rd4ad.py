"""
RD4AD (Reverse Distillation) - Anomaly detection su singole api (single bee crops).

Addestramento: immagini normali di singole api (VarroaDataset, split TRAIN),
               con augmentation geometrica (flip) perché le api possono avere
               qualsiasi orientamento.
Valutazione:   metriche a livello immagine e pixel (AUROC, AvgPrec, F1, AUPRO)
               sul test set sintetico (api sane + api con varroa + maschere).
Visualizzazione: heatmap normalizzate GLOBALMENTE (stessa scala per tutte le
                 immagini di test) e ancorate alla soglia di decisione:
                   v_min  -> 0.0 (blu)   livello di fondo delle api sane
                   soglia -> 0.5         soglia pixel che massimizza l'F1
                   v_max  -> 1.0 (rosso) anomalie più forti del test set
                 Così un'ape sana resta blu e solo ciò che supera la soglia
                 diventa giallo/rosso (con la normalizzazione min/max per
                 immagine, invece, ogni ape avrebbe sempre un punto rosso).
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.rd4ad.rd4ad import RD4AD, RD4ADTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"

BACKBONE      = "wide_resnet50_2"
IMG_SIZE      = (256, 256)   # RD4AD richiede multipli di 32
EPOCHS        = 20
BATCH_SIZE    = 16
EVAL_INTERVAL = 5            # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = "./heatmaps_rd4ad_debug"
CHECKPOINT    = "./checkpoints/rd4ad_single_bee.pth"


# ──────────────────────────────────────────────────────────────────────
# Utilità
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class AugmentedTrainDataset(Dataset):
    """Applica flip casuali alle immagini (già normalizzate) del dataset di training."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.aug(self.base[idx])


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def train_rd4ad_single_bee():
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
    model = RD4AD(backbone_name=BACKBONE, input_size=IMG_SIZE).to(device)
    # l'encoder (teacher) è congelato: l'optimizer aggiorna solo bottleneck e decoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)

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
    train_dataset = AugmentedTrainDataset(VarroaDataset(dataset_args, "varroa", Split.TRAIN))
    test_dataset = VarroaDataset(dataset_args, "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    training_args = RD4ADTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE,
                                   evaluation_epoch_interval=EVAL_INTERVAL)
    training_args.init_train(model)

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training RD4AD single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_loader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello RD4AD salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="RD4AD single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
    )


if __name__ == "__main__":
    train_rd4ad_single_bee()
