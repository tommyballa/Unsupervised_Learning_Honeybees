"""
PatchCore — Anomaly detection su singole api (dataset Ceschi).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
               val: 322 con varroa + 1251 sane, per la best e le soglie F1;
               test_heldout: 957 con varroa + 5000 sane, valutato una volta;
               vedi make_val_split.py).
Addestramento: feature wide_resnet50_2 ImageNet congelate (layer2-3) delle api
               sane (VarroaDataset, TRAIN), MAX_TRAIN_PATCHES patch casuali in totale
               (stesso numero per immagine, prese sull'ape e non sullo sfondo
               nero), memory bank ridotta a MEMORY_BANK_SIZE patch con coreset
               greedy. Nessuna
               epoca da scegliere: la memory bank viene salvata in
               checkpoints/single_bee/patchcore/ (best.pth + config.json).
Valutazione:   mappa = distanza dal vicino più prossimo nella memory bank
               (blur gaussiano MAP_SIGMA px), azzerata fuori dall'ape; score
               immagine = massimo della mappa.
Ottimizzazione (sweep: experiments/single_bee/sweep_patchcore.py, dataset
batch3+4, AUROC immagine a 512 px, bank 30k, 4M patch candidate):
  - configurazione precedente (layer1-3, blur sigma 2, score = media ripesata
    delle 5 patch peggiori): 0.928 (0.932 con 10M patch candidate);
  - blur della mappa più largo e score = massimo della mappa: sigma 8 -> 0.949,
    oltre sigma 12 peggiora (a 16-24 cala anche l'AP pixel): la varroa è una
    macchia di ~25 px, i falsi positivi picchi isolati di poche patch;
  - layer2-3 al posto di layer1-3: 0.956, inferenza in meno della metà del tempo;
  - memory bank 60k: +0.001, non vale il costo;
  - mobilenet_v2 (features.6 + features.13, 20k immagini di training): 0.963
    a livello immagine ma AP pixel più bassa (0.60 contro 0.66 a 128 px).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO), con la stessa
normalizzazione globale degli altri modelli:
    python experiments/plot_heatmaps.py single_bee patchcore

Il training usa PatchCoreCL (un solo task) come nella versione originale.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models import PatchCore
from moviad.models.patchcore.anomaly_map import AnomalyMapGenerator
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.strategies.patchcore_cl import PatchCoreCL
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "patchcore"
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATASET_DIR  = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR    = f"{DATASET_DIR}/train/normal"
VAL_DIR      = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR     = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE      = (512, 512)

# backbone ImageNet congelato e layer delle feature (stride 8 e 16). Il default
# è wide_resnet50_2; per confrontare i backbone:
#     PATCHCORE_BACKBONE=mobilenet_v2 python experiments/single_bee/main_patchcore.py
#     PATCHCORE_BACKBONE=mobilenet_v2 python experiments/plot_heatmaps.py single_bee patchcore
# con un backbone diverso dal default checkpoint e heatmap vanno in
# patchcore_<backbone>/, senza toccare quelli del default
BACKBONE_LAYERS = {
    "wide_resnet50_2": ["layer2", "layer3"],
    "resnet18":        ["layer2", "layer3"],
    "mobilenet_v2":    ["features.6", "features.13"],
    "efficientnet_b5": ["features.3", "features.5"],
}
DEFAULT_BACKBONE = "wide_resnet50_2"
BACKBONE     = os.environ.get("PATCHCORE_BACKBONE") or ablation.backbone(DEFAULT_BACKBONE)
LAYERS       = BACKBONE_LAYERS[BACKBONE]
RUN_NAME     = MODEL_NAME if BACKBONE == DEFAULT_BACKBONE else f"{MODEL_NAME}_{BACKBONE}"
MEMORY_BANK_SIZE = 30000
# patch candidate al coreset in totale (fp16 su CPU, ~31GB): ~100 per immagine
MAX_TRAIN_PATCHES = 10_000_000
MAP_SIGMA    = 8            # px a IMG_SIZE, blur gaussiano della mappa
# maschera dell'ape: pixel non neri (lo sfondo dei crop è nero, ~86% dell'immagine),
# chiusura per riempire le zone scure del corpo, poi margine attorno all'ape
FG_THRESHOLD = 10 / 255
FG_CLOSING   = 15           # px (lato del kernel)
FG_DILATION  = 8            # px di margine
IMG_SIZE     = (512, 512)
BATCH_SIZE   = 32
TEST_BATCH_SIZE = 16
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # metà sane, metà con varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee_val" / RUN_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / RUN_NAME
# ABL_TRAIN_N / ABL_BACKBONE: vedi ablation.py
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Modello e dati (usati anche da experiments/plot_heatmaps.py)
# ──────────────────────────────────────────────────────────────────────
def dataset_args(test_dir=TEST_DIR):
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        test_path=test_dir,
        img_size=IMG_SIZE,
        gt_mask_size=GT_SIZE,
        image_transform_list=[
            transforms.ToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )


def _dilate(mask, size):
    return F.max_pool2d(mask, size, stride=1, padding=size // 2)


def bee_foreground(images):
    """Maschera (B,1,H,W) dell'ape da immagini normalizzate ImageNet."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    rgb = images * std + mean
    mask = (rgb.amax(dim=1, keepdim=True) > FG_THRESHOLD).float()
    mask = 1 - _dilate(1 - _dilate(mask, FG_CLOSING), FG_CLOSING)    # chiusura
    return _dilate(mask, 2 * FG_DILATION + 1)


class PatchCoreSingleBee(PatchCoreCL):
    """PatchCoreCL con score immagine = massimo della mappa (smussata e azzerata fuori dall'ape)."""

    def forward(self, batch):
        anomaly_maps, _ = super().forward(batch)
        return anomaly_maps, anomaly_maps.flatten(1).amax(1)


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = PatchCore(feature_extractor=feature_extractor, memory_bank_size=MEMORY_BANK_SIZE)
    model.anomaly_map_generator = AnomalyMapGenerator(sigma=MAP_SIGMA)
    return PatchCoreSingleBee(model, max_train_patches=MAX_TRAIN_PATCHES, foreground_fn=bee_foreground)


def model_state(patchcore):
    return {"memory_bank": patchcore.vad_model.memory_bank[0].detach().cpu()}


def load_model_state(patchcore, state):
    patchcore.vad_model.memory_bank = {0: state["memory_bank"]}


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_patchcore_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    patchcore = build_model(device)
    train_dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))

    # memory bank (un solo task)
    training_args = TrainingArgs(batch_size=BATCH_SIZE, epochs=1)
    patchcore.train()
    patchcore.start_task(0, train_dataset, training_args)
    patchcore.train_task(0, train_dataset, None, metrics=[], device=device, train_args=training_args)

    save_checkpoint(CHECKPOINT_DIR, model_state(patchcore), script_config(globals()),
                    best_epoch=None, note="memory bank costruita in un solo passaggio")
    print(f"\nMemory bank PatchCore salvata in {CHECKPOINT_DIR}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_patchcore_single_bee()
