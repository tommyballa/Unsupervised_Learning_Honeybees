"""
CFA (Coupled-hypersphere-based Feature Adaptation) - Anomaly detection su
singole api (single bee crops).

Addestramento: backbone wide_resnet50_2 ImageNet congelato (layer1 + layer2),
               memory bank inizializzata con le feature adattate delle api
               sane e ridotta con KMeans (gamma_c), patch descriptor
               addestrato con la soft-boundary loss (AdamW lr 1e-3,
               wd 5e-4, amsgrad, 30 epoche come nel paper).
               Score immagine = massimo della heatmap smussata (paper).
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/cfa/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee cfa
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.cfa.cfa import CFA, CFATrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "cfa"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

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
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = CFA(
        feature_extractor=feature_extractor,
        backbone=BACKBONE,
        gamma_c=GAMMA_C,
        gamma_d=GAMMA_D,
        feature_maps_channels=FEATURE_CHANNELS,
    )
    model.to(device)
    return model


def model_state(model):
    # patch descriptor, raggio r e memory bank (il backbone congelato non è nello state dict)
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    # in un modello nuovo la memory bank non esiste ancora (viene creata all'epoca 0)
    model.memory_bank = torch.nn.Parameter(state["memory_bank"].clone().to(model.device),
                                           requires_grad=False)
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
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

    model = build_model(device)

    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)

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
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    # all'epoca 0 train_epoch inizializza la memory bank
    print(f"\n{'='*60}")
    print(f"Training CFA single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = model.train_epoch(epoch, train_loader, training_args)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest CFA — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_cfa_single_bee()
