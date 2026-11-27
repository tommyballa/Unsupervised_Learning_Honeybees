"""
STFPM (Student-Teacher Feature Pyramid Matching) - Anomaly detection su
singole api (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
               val: 322 con varroa + 1251 sane, per la best e le soglie F1;
               test_heldout: 957 con varroa + 5000 sane, valutato una volta;
               vedi make_val_split.py).
               Ri-ottimizzato su questo dataset (sweep_stfpm.py, run sam3_*,
               metriche dello sweep: pixel AP a 128 px). La ricetta del dataset
               merged (layer2+layer3, coseno su 1448 step) si ferma a img AUROC
               0.967 / pixel AP 0.41 con picco a 543 step e poi cala. Qui lo
               student generalizza ancora prima e layer2 diventa rumoroso:
                 - mappa dal solo layer3 (MAP_LAYERS = [2]), blur sigma 8 px:
                   0.981-0.983 / 0.55-0.57 tra 300 e 420 step (layer2+layer3:
                   0.974 / 0.48), score = massimo (top-k: +0.001);
                 - 600 step con coseno (0.978-0.983 da 300 step alla fine);
                 - lr 0.1 su 1200 step: peggio e instabile (picco 0.980 / 0.47).
               Con questo script (metriche ufficiali a 512): step 540 img_roc_auc
               0.983 / pxl_avg_prec 0.554 (0.97-0.98 da 300 step in poi).
               Sotto, la storia dello sweep sul dataset merged (23k api).
Addestramento: teacher resnet18 ImageNet congelato, student resnet18 con pesi
               casuali addestrato a riprodurre le feature di layer1-3 del
               teacher sulle api sane (VarroaDataset, TRAIN). Ottimizzatore del
               paper (SGD lr 0.4, momentum 0.9, weight decay 1e-4, batch 32)
               con cosine annealing per step, bf16 autocast in training.
Ottimizzazione (sweep: experiments/single_bee/sweep_stfpm.py, logs/stfpm_sweep/;
metriche dello sweep: pixel AP a 128 px):
  - risoluzione nativa 512 (la varroa ha diametro ~33 px): dopo 1 epoca
    img AUROC ~0.90 a 256 (massimo 0.917 in 2.5 epoche) contro ~0.97-0.98 a 512;
  - mappa = somma di layer2 e layer3 (MAP_LAYERS) con blur gaussiano
    (MAP_SIGMA): layer1 è rumoroso. Il prodotto dei 3 layer senza blur del
    paper si ferma a img AUROC 0.947 / pixel AP 0.20 quando layer2+layer3 con
    blur dà 0.983 / 0.53. Somma e prodotto di layer2-3 sono equivalenti per
    l'AUROC, la somma è migliore per la pixel AP; lo score top-k e la
    normalizzazione per layer non aiutano. Il solo layer3 ha pixel AP un po'
    più alta (~0.58 contro ~0.55) ma img AUROC più bassa (~0.976);
  - training breve: lo student generalizza presto anche sulla varroa. Il picco
    è tra 1 e 1.5 epoche; con 8 epoche img AUROC scende a 0.93 e pixel AP a
    0.27 mentre la loss continua a calare. 2 epoche con coseno (0.982-0.985
    da 1 epoca in poi) sono più stabili di 4 (0.975-0.983), quindi valutazione
    ogni 1/4 di epoca;
  - non aiutano: flip + rot90 in training (picco 0.980), layer4 come layer
    aggiuntivo dello student (0.982, e la mappa migliore resta layer2+layer3).
Con questo script (metriche ufficiali a 512): step 905 (1.25 epoche)
img_roc_auc 0.985 / pxl_avg_prec 0.553, e 0.981-0.985 da 3/4 di epoca in poi.
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul validation set;
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.stfpm.stfpm import STFPM
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "stfpm"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

# layer a stride 4, 8 e 16; backbone diversi solo per l'ablation (ABL_BACKBONE, vedi ablation.py)
BACKBONE_LAYERS = {
    "resnet18":        ["layer1", "layer2", "layer3"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "mobilenet_v2":    ["features.3", "features.6", "features.13"],
}
BACKBONE      = ablation.backbone("resnet18")
LAYERS        = BACKBONE_LAYERS[BACKBONE]
IMG_SIZE      = (512, 512)   # risoluzione nativa dei crop
MAP_LAYERS    = [2]          # indici in LAYERS dei layer usati per la mappa (solo layer3; merged: [1, 2])
MAP_AGGREGATION = "sum"      # il paper usa "prod" su tutti i layer
MAP_SIGMA     = 8.0          # blur gaussiano della mappa (px a 512) prima dello score
TRAIN_STEPS   = 600          # coseno su 600 step: il picco è tra 300 e 420 step (merged: 1448)
BATCH_SIZE    = 32
LR            = 0.4          # come nel paper STFPM (SGD, momentum 0.9), coseno per step fino a 0
WEIGHT_DECAY  = 1e-4
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 60           # valuta ogni N step e sempre alla fine
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


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
def dataset_args(test_dir=TEST_DIR):
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
        test_path=test_dir,
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
    return STFPM(teacher, student, map_aggregation=MAP_AGGREGATION,
                 map_layers=MAP_LAYERS, map_sigma=MAP_SIGMA).to(device)


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

    train_dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # SGD sui soli parametri dello student, cosine annealing per step
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    total_steps = TRAIN_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training STFPM single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epoche "
          f"| {len(train_loader)} batch/epoca | lr {LR} | valutazione ogni {EVAL_INTERVAL} step")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            model.train()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                teacher_features, student_features = model(batch)
            loss = 0.0
            for t, s in zip(teacher_features, student_features):
                t = F.normalize(t.float(), dim=1)
                s = F.normalize(s.float(), dim=1)
                loss = loss + (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoca {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(model, val_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(model))
            if step >= total_steps:
                break

    print(f"\nBest STFPM (\"epoca\" = step di training) — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_stfpm_single_bee()
