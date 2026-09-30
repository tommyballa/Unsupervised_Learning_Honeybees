"""
PaDiM - Anomaly detection su singole api (single bee crops).

Addestramento: stima di una gaussiana multivariata per posizione sulle feature
               (backbone ImageNet congelato) delle api sane (VarroaDataset, TRAIN).
               Un solo passaggio sui dati, covarianza con np.cov + 0.01 I come
               nel paper. Nessuna epoca da scegliere: media e covarianze
               stimate vengono salvate in checkpoints/single_bee/padim/
               (best.pth + config.json).
Valutazione:   distanza di Mahalanobis per posizione, upsampling + gaussiano
               (sigma 4), score immagine = massimo della mappa.
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO):
    python experiments/plot_heatmaps.py single_bee padim

Backbone mobilenet_v2 a 224 px: con wide_resnet50_2 le feature di ~10k api
occuperebbero decine di GB di RAM (PaDiM le tiene tutte in memoria).
"""

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
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "padim"
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
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
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
    return Padim(backbone_model_name=BACKBONE, layers_idxs=LAYERS_IDXS, device=device).to(device)


def model_state(model):
    return {
        "gauss_mean": model.gauss_mean,
        "gauss_cov": model.gauss_cov,
        "random_dimensions": model.random_dimensions.detach().cpu(),
    }


def load_model_state(model, state):
    model.gauss_mean = state["gauss_mean"]
    model.gauss_cov = state["gauss_cov"]
    # le dimensioni della random projection devono essere quelle usate nel fitting
    model.random_dimensions = torch.nn.Parameter(state["random_dimensions"], requires_grad=False)
    model.inv_cov_tensor = None


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
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

    model = build_model(device)
    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\nFitting PaDiM ({BACKBONE}, layer {LAYERS_IDXS}) su {len(train_dataset)} api sane...")
    model.train()
    model.train_epoch(0, train_loader, PadimTrainArgs(epochs=1, batch_size=BATCH_SIZE))

    save_checkpoint(CHECKPOINT_DIR, model_state(model), script_config(globals()),
                    best_epoch=None, note="fitting in un solo passaggio (nessuna epoca da scegliere)")
    print(f"Gaussiane PaDiM salvate in {CHECKPOINT_DIR}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_padim_single_bee()
