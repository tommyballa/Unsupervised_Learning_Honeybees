"""
SuperSimpleNet - Anomaly detection su singole api (single bee crops).

Addestramento: immagini normali di singole api (VarroaDataset, split TRAIN).
               Le anomalie vengono simulate nello spazio delle feature
               (rumore gaussiano mascherato con Perlin noise), come nel paper.
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
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, evaluate_and_report, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE         = (512, 512)

BACKBONE        = "wide_resnet50_2"
LAYERS          = ["layer2", "layer3"]   # come nel paper SuperSimpleNet
IMG_SIZE        = (256, 256)
EPOCHS          = 30
BATCH_SIZE      = 32
EVAL_INTERVAL   = 5            # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS     = 8
SEED            = 1

# Iperparametri del paper / anomalib
ADAPTOR_LR      = 1e-4
SEGDEC_LR       = 2e-4
WEIGHT_DECAY    = 1e-5
LR_GAMMA        = 0.4
LR_MILESTONES   = [int(EPOCHS * 0.8), int(EPOCHS * 0.9)]   # in epoche
PERLIN_THRESHOLD = SuperSimpleNet.DEFAULT_PARAMETERS["perlin_threshold"]

NUM_HEATMAPS    = 20           # metà sane, metà con varroa
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee" / "supersimplenet"
CHECKPOINT      = REPO_ROOT / "checkpoints" / "single_bee" / "supersimplenet.pth"


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
# Training
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    avg_loss = 0.0
    for images in tqdm(dataloader):
        images = images.to(device)
        B, _, H, W = images.shape
        # tutte le immagini di training sono normali: maschere e label a zero,
        # le anomalie vengono generate dal modello nello spazio delle feature
        masks = torch.zeros(B, 1, H, W, device=device)
        labels = torch.zeros(B, device=device)

        anomaly_map, anomaly_score, masks, labels = model(images=images, masks=masks, labels=labels)
        loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score,
                       target_mask=masks, target_label=labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    return avg_loss / len(dataloader)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def train_supersimplenet_single_bee():
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
    model = SuperSimpleNet(
        feature_extractor=feature_extractor,
        perlin_threshold=PERLIN_THRESHOLD,
        stop_grad=SuperSimpleNet.DEFAULT_PARAMETERS["stop_grad"],
    ).to(device)

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

    # --- Optimizer, scheduler (per epoca) e loss ---
    optimizer = AdamW([
        {"params": model.adaptor.parameters(), "lr": ADAPTOR_LR},
        {"params": model.segdec.parameters(), "lr": SEGDEC_LR, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)
    loss_fn = SSNLoss()

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Training SuperSimpleNet single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()}")

        if (epoch + 1) % EVAL_INTERVAL == 0 and epoch != EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            print_metrics(compute_metrics(maps, scores, labels, masks, with_pro=False))

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\nModello SuperSimpleNet salvato in {CHECKPOINT}")

    # --- Valutazione finale + heatmap normalizzate ---
    evaluate_and_report(
        model, test_dataset, test_loader, device, OUTPUT_DIR,
        title="SuperSimpleNet single-bee", num_heatmaps=NUM_HEATMAPS, seed=SEED,
        raw_label="Mappa grezza (logit)",
    )


if __name__ == "__main__":
    train_supersimplenet_single_bee()
