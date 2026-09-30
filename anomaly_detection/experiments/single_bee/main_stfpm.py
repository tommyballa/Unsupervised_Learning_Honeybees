"""
STFPM (Student-Teacher Feature Pyramid Matching) - Anomaly detection su
singole api (single bee crops).

Addestramento: teacher resnet18 ImageNet congelato, student resnet18 con pesi
               casuali addestrato a riprodurre le feature di layer1-3 del
               teacher sulle api sane (VarroaDataset, TRAIN). Ricetta del paper:
               SGD lr 0.4, momentum 0.9, weight decay 1e-4, batch 32, 256 px.
               Con ~10k api, 20 epoche sono ~6k iterazioni (il paper ne fa ~1k
               su MVTec).
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/stfpm/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee stfpm

Per l'esperimento continual (domain shift, replay) c'è main_stfpm_clad.py.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "stfpm"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

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
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Modello e dati (usati anche da experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def dataset_args():
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_model(device):
    # teacher pre-addestrato congelato, student con pesi casuali
    teacher = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    student = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=False)
    return STFPM(teacher, student).to(device)


def model_state(model):
    # il teacher è il resnet18 ImageNet: basta lo student
    return clean_state_dict(model.student.model.state_dict())


def load_model_state(model, state):
    model.student.model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
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

    model = build_model(device)

    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)

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
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training STFPM single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca | lr {LR}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        loss = model.train_epoch(epoch, train_loader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest STFPM — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_stfpm_single_bee()
