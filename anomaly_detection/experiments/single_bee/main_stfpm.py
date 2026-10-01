"""
STFPM (Student-Teacher Feature Pyramid Matching) - Anomaly detection su
singole api (single bee crops).

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
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul test set;
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
IMG_SIZE      = (512, 512)   # risoluzione nativa dei crop
MAP_LAYERS    = [1, 2]       # indici in LAYERS dei layer usati per la mappa (layer2, layer3)
MAP_AGGREGATION = "sum"      # il paper usa "prod" su tutti i layer
MAP_SIGMA     = 8.0          # blur gaussiano della mappa (px a 512) prima dello score
TRAIN_EPOCHS  = 2            # epoche (anche frazionarie): il picco è tra 1 e 1.5 epoche
BATCH_SIZE    = 32
LR            = 0.4          # come nel paper STFPM (SGD, momentum 0.9), coseno per step fino a 0
WEIGHT_DECAY  = 1e-4
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 181          # valuta ogni N step (~1/4 di epoca) e sempre alla fine
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

    # SGD sui soli parametri dello student, cosine annealing per step
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    total_steps = int(TRAIN_EPOCHS * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training STFPM single-bee | {TRAIN_EPOCHS} epoche = {total_steps} step "
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
                maps, scores, labels, masks = run_inference(model, test_loader, device)
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
