"""
PaSTe (Partially Shared Teacher-student) - Anomaly detection su singole api
(single bee crops).

Addestramento: teacher resnet18 ImageNet congelato; lo student (pesi casuali)
               parte dall'output di conv1 del teacher (bootstrap layer 0) e
               impara a riprodurre le feature di layer1-3 sulle api sane
               (VarroaDataset, TRAIN). SGD lr 0.4 (momentum 0.9, weight decay
               1e-4) con cosine annealing per step su TRAIN_STEPS step, bf16
               autocast; la loss usa MSE con reduction="sum". Input 512x512
               (risoluzione nativa dei crop).
Ottimizzazione (sweep: experiments/single_bee/sweep_paste.py, logs/paste_sweep/;
dataset batch3+4: 68.5k api di train, test 5000 sane + 3119 con varroa;
metriche dello sweep: img AUROC / pixel AP a 128 px, miglior variante di mappa):
  - lo student generalizza presto anche sulla varroa: le metriche hanno il
    picco dopo poche centinaia di step e poi scendono mentre la loss continua
    a calare. Con resnet18 e lr 0.4 il picco è a ~250 step (0.944 / 0.43) e
    dopo 2000 step si scende a 0.906 / 0.28; con 400 step e coseno si resta a
    0.939-0.945 / 0.42-0.47 da 250 step in poi, quindi training di 400 step
    con valutazione ogni 50;
  - resnet18 al posto di wide_resnet50_2: la ricetta precedente (wide_resnet50_2,
    lr 0.04, mappa dal solo layer3) si ferma a 0.89-0.90 / 0.21-0.26 in 3000
    step, e 0.90-0.92 con lr 0.4;
  - mappa = somma di layer2 e layer3 con blur gaussiano (MAP_SIGMA 16 px a
    512): layer1 è rumoroso (0.92 con i 3 layer), senza blur o con blur 32 px
    si perde (0.92). Il solo layer3 ha pixel AP un po' più alta (~0.45 contro
    ~0.42) ma img AUROC appena più bassa; normalizzazione per layer, prodotto
    e score top-k sono equivalenti;
  - non aiutano: loss solo sulle celle dell'ape (0.92-0.93; azzerare la mappa
    sullo sfondo nero non cambia nulla), lr 0.1 (0.934-0.940), bootstrap al
    layer1 (0.90-0.92), input ingrandito a 768 (0.92) o 1024 (0.87). Senza
    bootstrap (student dall'immagine, cioè STFPM) il risultato è lo stesso
    (0.945-0.948): condividere conv1 non costa accuratezza. Sul dataset
    precedente non aiutavano nemmeno EMA dei pesi e flip/rot90.
Con questo script (metriche ufficiali a 512): step 400 img_roc_auc 0.932 /
pxl_avg_prec 0.404, e 0.931-0.932 / 0.40-0.42 da 250 step in poi.
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul validation set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/paste/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee paste
"""

from moviad.datasets.varroa.paths import DATA_ROOT
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


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "paste"
REPO_ROOT       = Path(__file__).resolve().parents[2]
DATASET_DIR     = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR       = f"{DATASET_DIR}/train/normal"
VAL_DIR         = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR        = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE         = (512, 512)

# indici dei blocchi a stride 4, 8 e 16 (AD) e 8, 16 (mappa); backbone diversi
# solo per l'ablation (ABL_BACKBONE, vedi ablation.py)
BACKBONE_LAYERS = {
    "resnet18":        ([1, 2, 3], [2, 3]),
    "wide_resnet50_2": ([1, 2, 3], [2, 3]),
    "mobilenet_v2":    ([3, 6, 13], [6, 13]),
}
BACKBONE        = ablation.backbone("resnet18")
AD_LAYERS, MAP_LAYERS = BACKBONE_LAYERS[BACKBONE]
BOOTSTRAP_LAYER = 0
TEACHER_WEIGHTS = "IMAGENET1K_V1"
IMG_SIZE        = (512, 512)    # risoluzione nativa dei crop
MAP_AGGREGATION = "sum"
MAP_SIGMA       = 16.0        # blur gaussiano della mappa (px a 512) prima dello score
TRAIN_STEPS     = 400        # il picco è a ~250 step: training corto con coseno fino a 0
BATCH_SIZE      = 32
LR              = 0.4
AMP             = True       # bf16 autocast in training
EVAL_INTERVAL   = 50         # valuta ogni N step (e sempre alla fine)
NUM_WORKERS     = 8
SEED            = 1

NUM_HEATMAPS    = 20         # metà sane, metà con varroa
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR  = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
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
    # il teacher è il backbone ImageNet: basta lo student
    return clean_state_dict(model.student.state_dict())


def load_model_state(model, state):
    model.student.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_step(model, images, optimizer, scheduler, device):
    """Come PaSTe.train_step, ma con bf16 autocast e scheduler per step."""
    model.train()
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
    return loss.item()


def infinite(loader):
    while True:
        yield from loader


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

    # SGD (momentum 0.9, weight decay 1e-4) sui soli parametri dello student
    training_args = PasteTrainArgs(epochs=1, batch_size=BATCH_SIZE, lr=LR)
    training_args.init_train(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(training_args.optimizer, T_max=TRAIN_STEPS)
    # "epoca" della best = step di training
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training PaSTe single-bee | {TRAIN_STEPS} step | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    batches = infinite(train_loader)
    total, n = 0.0, 0
    for step in tqdm(range(1, TRAIN_STEPS + 1), desc="Training"):
        total += train_step(model, next(batches), training_args.optimizer, scheduler, device)
        n += 1

        if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
            print(f"\nStep {step}/{TRAIN_STEPS} — loss: {total / n:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")
            total, n = 0.0, 0
            maps, scores, labels, masks = run_inference(model, val_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(step, results, lambda: model_state(model))

    print(f"\nBest PaSTe (\"epoca\" = step di training) — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_paste_single_bee()
