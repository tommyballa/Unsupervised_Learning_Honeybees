"""
RD4AD (Reverse Distillation) - Anomaly detection su singole api (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
               val: 322 con varroa + 1251 sane, per la best e le soglie F1;
               test_heldout: 957 con varroa + 5000 sane, valutato una volta;
               vedi make_val_split.py).
               Ri-ottimizzato su questo dataset (sweep_rd4ad.py, run sam3_*,
               metriche dello sweep: pixel AP a 128 px). La ricetta del dataset
               merged (724 step, layer 1+2, sigma 4) si ferma a img AUROC 0.914 /
               pixel AP 0.27 (picco a 273 step, poi ~0.88):
                 - 400 step con coseno, valutazione ogni 40: il picco è tra 160 e
                   400 step ma oscilla di ±0.02 tra valutazioni vicine;
                 - mappa dal solo layer 2 con sigma 8 px (= 4 a 256): ~0.94-0.95 /
                   0.33-0.37 (layer 1+2: ~0.94 / 0.35-0.38; layer 0+1+2: 0.86);
                   sigma 12, 16 o 24 px e score top-k non superano ~0.945;
                 - lr 0.001: picco precoce (0.943 a 160 step) e poi calo a 0.92.
               Con questo script (metriche ufficiali a 512): step 400 img_roc_auc
               0.953 / pxl_avg_prec 0.362, AU-PRO 0.989 (0.93-0.95 da 160 step).
               Sotto, la storia dello sweep sul dataset merged (23k api).
Addestramento: immagini normali di singole api (VarroaDataset, split TRAIN),
               con augmentation geometrica (flip) perché le api possono avere
               qualsiasi orientamento.
Ottimizzazione (sweep: experiments/single_bee/sweep_rd4ad.py):
  - risoluzione nativa 512 (le varroe sono piccole): img AUROC ~0.90 a 256,
    ~0.96 a 384, ~0.978 a 512;
  - mappa dai soli layer 1 e 2 (il layer 0 è rumoroso: img AP 0.85 contro ~0.75);
  - training breve: con 23k immagini RD4AD over-generalizza già dopo ~1 epoca
    (il decoder impara a ricostruire anche le anomalie e le metriche calano),
    quindi poche epoche con LR coseno e valutazione frequente;
  - EMA dei pesi (EMA_DECAY) disponibile ma disattivata: stabilizza le run
    lunghe, ma al picco precoce è in ritardo (384: img AUROC 0.954 contro 0.962);
  - training in bf16 (autocast), valutazione in fp32.
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul validation set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/rd4ad/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee rd4ad
"""

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.rd4ad.loss_functions import rd4ad_cosine_loss
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "rd4ad"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

BACKBONE      = ablation.backbone("wide_resnet50_2")   # resnet18: ablation (ABL_BACKBONE)
IMG_SIZE      = (512, 512)   # RD4AD richiede multipli di 32
TRAIN_STEPS   = 400          # step totali: picco tra 160 e 400 step (merged: 724 = 1/2 epoca)
BATCH_SIZE    = 16
LR            = 0.005        # paper: Adam lr 0.005, betas (0.5, 0.999)
BETAS         = (0.5, 0.999)
WARMUP_FRAC   = 0.02         # warmup lineare, poi decadimento coseno fino a 0
EMA_DECAY     = 0.0          # EMA dei pesi di bn + decoder (0 = off: al picco precoce l'EMA è in ritardo)
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 40           # valuta ogni N step e sempre alla fine
BLUR_SIGMA    = 8.0          # sigma del blur sulla mappa a IMG_SIZE (= 4 a 256; merged: 4)
LAYERS        = (2,)         # layer sommati nella mappa (None = tutti e 3; merged: (1, 2))
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────
# Utilità
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class AugmentedTrainDataset(Dataset):
    """Applica flip casuali alle immagini (già normalizzate) del dataset di training."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.aug(self.base[idx])


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
    model = RD4AD(backbone_name=BACKBONE, input_size=IMG_SIZE,
                  blur_sigma=BLUR_SIGMA, layers=LAYERS).to(device)
    # l'encoder (teacher) è congelato: l'optimizer aggiorna solo bottleneck e decoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


def model_state(model):
    # l'encoder è il wide_resnet50_2 ImageNet congelato: bastano bottleneck e decoder
    return {"bn": clean_state_dict(model.bn.state_dict()),
            "decoder": clean_state_dict(model.decoder.state_dict())}


def load_model_state(model, state):
    model.bn.load_state_dict(state["bn"])
    model.decoder.load_state_dict(state["decoder"])


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def update_ema(ema_model, model, step):
    # decay ridotto nei primi step (come timm ModelEmaV2 con warmup)
    decay = min(EMA_DECAY, (1 + step) / (10 + step))
    for ema_mod, mod in ((ema_model.bn, model.bn), (ema_model.decoder, model.decoder)):
        for e, m in zip(ema_mod.state_dict().values(), mod.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(m.detach(), alpha=1 - decay)
            else:
                e.copy_(m)


def train_rd4ad_single_bee():
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
    ema_model = copy.deepcopy(model) if EMA_DECAY > 0 else model
    ema_model.encoder = model.encoder  # encoder congelato condiviso

    train_dataset = AugmentedTrainDataset(ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN)))
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # l'encoder (teacher) è congelato: l'optimizer aggiorna solo bottleneck e decoder
    optimizer = torch.optim.Adam(list(model.decoder.parameters()) + list(model.bn.parameters()),
                                 lr=LR, betas=BETAS)
    total_steps = TRAIN_STEPS
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\n{'='*60}")
    print(f"Training RD4AD single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epoche "
          f"| {len(train_loader)} batch/epoca | valutazione ogni {EVAL_INTERVAL} step")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            model.train()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                teacher = model.encoder(batch)
                student = model.decoder(model.bn(teacher))
            loss = sum(rd4ad_cosine_loss(t.float(), s.float()) for t, s in zip(teacher, student))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if ema_model is not model:
                update_ema(ema_model, model, step)
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoca {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(ema_model, val_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(ema_model))
            if step >= total_steps:
                break

    print(f"\nBest RD4AD (\"epoca\" = step di training) — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_rd4ad_single_bee()
