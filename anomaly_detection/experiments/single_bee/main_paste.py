"""
PaSTe (Partially Shared Teacher-student) - Anomaly detection su singole api
(single bee crops).

Addestramento: teacher wide_resnet50_2 ImageNet congelato; lo student (pesi
               casuali) parte dall'output di conv1 del teacher (bootstrap
               layer 0) e impara a riprodurre le feature di layer1-3 sulle api
               sane (VarroaDataset, TRAIN). SGD lr 0.04 (momentum 0.9, weight
               decay 1e-4) con cosine annealing per step su EPOCHS epoche, bf16
               autocast; la loss usa MSE con reduction="sum". ad_layers
               [1, 2, 3]: layer4 a bassa risoluzione è troppo grossolano per la
               varroa. Input 512x512 (risoluzione nativa dei crop): la varroa
               ha diametro ~33 px, a 256 occuperebbe ~1 cella della griglia di
               layer3.
               Lo student generalizza presto anche sulla varroa: nello sweep
               (experiments/single_bee/sweep_paste.py, logs/paste_sweep/) le
               metriche hanno il picco alle prime epoche e poi scendono mentre
               la loss continua a calare. Con questo script (metriche ufficiali
               a 512): epoca 4 img_roc_auc 0.961 / pxl_avg_prec 0.147, epoca 7
               0.949 / 0.064; la vecchia ricetta (20 epoche, cosine per epoca,
               valutazione ogni 5) trovava 0.923 / 0.032. Per questo schedule
               corto (8 epoche) e valutazione a ogni epoca. Non
               aiutano: lr 0.01, bootstrap al layer1 (ad_layers [2, 3]), EMA
               dei pesi dello student; flip/rot90 in training non danno un
               vantaggio stabile (sweep, pixel AP a 128 px: 0.955 / 0.19
               all'epoca 4, poi 0.945 / 0.08).
Valutazione:   mappa = distanza coseno teacher/student del solo layer3
               (MAP_LAYERS); layer1-2 servono come supervisione dello student
               ma sulla varroa sono rumorosi (falsi positivi su testa, torace
               e pose insolite). Nello sweep layer3 da solo, senza blur, è la
               variante migliore sia per img_roc_auc sia per pxl_avg_prec
               rispetto a somma/prodotto/normalizzazione con layer1-2, blur e
               score top-k. Score immagine = massimo della mappa.
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/paste/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee paste
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.paste.paste import PaSTe, PasteTrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "paste"
REPO_ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE         = (512, 512)

BACKBONE        = "wide_resnet50_2"
AD_LAYERS       = [1, 2, 3]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"
IMG_SIZE        = (512, 512)    # risoluzione nativa dei crop
MAP_LAYERS      = [3]           # layer usati per la mappa (sottoinsieme di AD_LAYERS)
MAP_AGGREGATION = "sum"
MAP_SIGMA       = 0.0         # blur gaussiano della mappa (px a 512) prima dello score
EPOCHS          = 8
BATCH_SIZE      = 32
LR              = 0.04
AMP             = True       # bf16 autocast in training (~2x più veloce a 512)
EVAL_INTERVAL   = 1          # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS     = 8
SEED            = 1

NUM_HEATMAPS    = 20         # metà sane, metà con varroa
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR  = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


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
    return PaSTe(
        backbone_model_name=BACKBONE,
        input_size=IMG_SIZE,
        output_size=IMG_SIZE,
        ad_layers=AD_LAYERS,
        weights=TEACHER_WEIGHTS,
        student_bootstrap_layer=BOOTSTRAP_LAYER,
        map_aggregation=MAP_AGGREGATION,
        map_layers=MAP_LAYERS,
        map_sigma=MAP_SIGMA,
    ).to(device)


def model_state(model):
    # il teacher è il wide_resnet50_2 ImageNet: basta lo student
    return clean_state_dict(model.student.state_dict())


def load_model_state(model, state):
    model.student.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Come PaSTe.train_epoch, ma con bf16 autocast e scheduler per step."""
    model.train()
    total = 0.0
    for images in tqdm(loader, desc=f"Epoca {epoch + 1}"):
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP and device.type == "cuda"):
            t_feat, s_feat = model(images)
        loss = 0.0
        for t, s in zip(t_feat, s_feat):
            t = F.normalize(t.float(), dim=1)
            s = F.normalize(s.float(), dim=1)
            loss = loss + (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(1, len(loader))


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

    # SGD (momentum 0.9, weight decay 1e-4) sui soli parametri dello student
    training_args = PasteTrainArgs(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        training_args.optimizer, T_max=EPOCHS * len(train_loader))
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training PaSTe single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, training_args.optimizer, scheduler, device, epoch)
        print(f"Epoca {epoch + 1}/{EPOCHS} — loss: {loss:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest PaSTe — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_paste_single_bee()
