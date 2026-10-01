"""
FastFlow - Anomaly detection su singole api (single bee crops).

Addestramento: feature wide_resnet50_2 ImageNet congelate (layer1-3) sui crop
               a risoluzione nativa 512x512, LayerNorm addestrabili per scala e un
               normalizing flow 2D per scala, massimizzando la likelihood delle api
               sane (VarroaDataset, TRAIN). Adam lr 3e-4, wd 1e-5, one-cycle cosine.
               Differenze rispetto al FastFlow standard (vedi sweep_fastflow.py):
               - i crop hanno sfondo nero: la loss è la NLL per pixel mediata solo
                 sui pixel dell'ape (altrimenti lo sfondo quasi costante domina la
                 loss e la mappa si accende su sfondo e bordi);
               - pochi step (STEPS): il flow raggiunge il picco dopo 50-150 step e poi
                 peggiora (impara anche la varroa come "probabile");
               - mappa = NLL per pixel (0.5 z^2 - log-det) sulle scale layer2+3,
                 limitata all'ape erosa di FG_ERODE px, smussata (sigma BLUR_SIGMA);
                 score immagine = massimo della mappa.
Best:          ogni EVAL_INTERVAL step (e all'ultimo) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/fastflow/ (best.pth +
               config.json). best.pth è lo state dict completo del modello, lo
               stesso formato letto da diagnose_fastflow_varroa.py.
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee fastflow
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.fastflow.fastflow import create_fastflow, FastFlowTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "fastflow"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

# cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2
BACKBONE      = "wide_resnet50_2"
IMG_SIZE      = (512, 512)   # risoluzione nativa: a 256 la varroa copre ~14x14 px
STEPS         = 200          # step di training (1 epoca = ~2900 step)
BATCH_SIZE    = 8
LR            = 3e-4
WEIGHT_DECAY  = 1e-5
EVAL_INTERVAL = 25           # valuta ogni N step (e sempre all'ultimo)
NUM_WORKERS   = 8

# foreground: crop delle api su sfondo nero
FG_THRESHOLD  = 0.1          # intensità RGB massima (0-1) sopra cui un pixel è ape
MASKED_LOSS_DILATE = 4       # loss sui pixel dell'ape dilatati di N px
FG_ERODE      = 8            # mappa limitata all'ape erosa di N px (bordo della sagoma instabile)
# mappa di anomalia
MAP_TYPE      = "nll"        # "z2" (FastFlow originale) | "nll"
MAP_SCALES    = (1, 2)       # layer2, layer3
BLUR_SIGMA    = 8.0
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True è sicuro con input size fisso e velocizza le convoluzioni
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
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
    # LayerNorm per scala + normalizing flow, come nel paper
    return create_fastflow(
        img_shape=IMG_SIZE, backbone_name=BACKBONE, device=device, normalize=True,
        blur_sigma=BLUR_SIGMA, map_type=MAP_TYPE, map_scales=MAP_SCALES,
        foreground_threshold=FG_THRESHOLD, foreground_erode=FG_ERODE,
        masked_loss_dilate=MASKED_LOSS_DILATE, backbone_amp=True,
        image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD,
    )


def model_state(model):
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
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

    # --- Optimizer (flow + LayerNorm), scheduler per step e loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=STEPS, pct_start=0.05,
        anneal_strategy="cos", div_factor=10, final_div_factor=100)
    training_args = FastFlowTrainArgs(epochs=1, batch_size=BATCH_SIZE,
                                      optimizer=optimizer, scheduler=scheduler)
    training_args.init_train(model)
    # la best è indicizzata per step (config.json: "best_epoch" = step)
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training FastFlow single-bee | {STEPS} step | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    def batches():
        while True:
            yield from train_loader

    model.train()
    running = 0.0
    for step, batch in enumerate(batches(), start=1):
        running += model.train_step(batch, training_args)
        scheduler.step()

        if step % EVAL_INTERVAL == 0 or step == STEPS:
            print(f"Step {step}/{STEPS} — loss: {running / EVAL_INTERVAL:.4f}, "
                  f"lr: {scheduler.get_last_lr()[0]:.2e}")
            running = 0.0
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            del maps
            print_metrics(results)
            tracker.update(step, results, lambda: model_state(model))
            model.train()
        if step >= STEPS:
            break

    print(f"\nBest FastFlow — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_fastflow_single_bee()
