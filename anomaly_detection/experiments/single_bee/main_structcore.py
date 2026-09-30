"""
StructCore - Anomaly detection su singole api (single bee crops).

Addestramento: come PatchCore (feature wide_resnet50_2 layer1 + layer2, 10%
               delle patch di ogni immagine, coreset greedy), più la
               calibrazione del descrittore strutturale (dispersione, media
               del top-1%, variazione totale della mappa) sulle api sane.
               Nessuna epoca da scegliere: memory bank e calibrazione vengono
               salvate in checkpoints/single_bee/structcore/ (best.pth +
               config.json).
Valutazione:   mappa = distanza dal vicino più prossimo nella memory bank,
               score immagine = distanza di Mahalanobis diagonale del
               descrittore strutturale (calcolato su tutta la mappa).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO):
    python experiments/plot_heatmaps.py single_bee structcore

Il training usa StructCoreCL (un solo task) per il sottocampionamento al 10%
delle patch, necessario per non esaurire la memoria.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import StructCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.structcore_cl import StructCoreCL
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "structcore"
REPO_ROOT    = Path(__file__).resolve().parents[2]
TRAIN_DIR    = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE      = (512, 512)

BACKBONE     = "wide_resnet50_2"
LAYERS       = ["layer1", "layer2"]
MEMORY_BANK_SIZE = 10000
IMG_SIZE     = (384, 384)
BATCH_SIZE   = 8
TEST_BATCH_SIZE = 16
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
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    return StructCoreCL(StructCore(feature_extractor=feature_extractor, memory_bank_size=MEMORY_BANK_SIZE))


def model_state(structcore):
    model = structcore.vad_model
    return {
        "memory_bank": model.memory_bank[0].detach().cpu(),
        "train_mu": model.train_mu[0].detach().cpu(),
        "train_sigma": model.train_sigma[0].detach().cpu(),
    }


def load_model_state(structcore, state):
    model = structcore.vad_model
    device = model.device
    model.memory_bank = {0: state["memory_bank"]}
    model.train_mu = {0: state["train_mu"].to(device)}
    model.train_sigma = {0: state["train_sigma"].to(device)}


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_structcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    structcore = build_model(device)
    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)

    # memory bank + calibrazione del descrittore (un solo task)
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    structcore.train()
    structcore.start_task(0, train_dataset, training_args)
    structcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    save_checkpoint(CHECKPOINT_DIR, model_state(structcore), script_config(globals()),
                    best_epoch=None, note="memory bank e calibrazione in un solo passaggio")
    print(f"\nMemory bank e calibrazione StructCore salvate in {CHECKPOINT_DIR}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_structcore_single_bee()
