"""
RD4AD (Reverse Distillation) - Anomaly detection su singole api (single bee crops).

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
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul test set;
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


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "rd4ad"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

BACKBONE      = "wide_resnet50_2"
IMG_SIZE      = (512, 512)   # RD4AD richiede multipli di 32
TRAIN_EPOCHS  = 0.5          # epoche (anche frazionarie): il picco è tra 1/4 e 1/2 epoca
BATCH_SIZE    = 16
LR            = 0.005        # paper: Adam lr 0.005, betas (0.5, 0.999)
BETAS         = (0.5, 0.999)
WARMUP_FRAC   = 0.02         # warmup lineare, poi decadimento coseno fino a 0
EMA_DECAY     = 0.0          # EMA dei pesi di bn + decoder (0 = off: al picco precoce l'EMA è in ritardo)
AMP           = True         # bf16 autocast in training
EVAL_INTERVAL = 181          # valuta ogni N step (~1/8 di epoca) e sempre alla fine
BLUR_SIGMA    = 4.0          # sigma del blur sulla mappa a IMG_SIZE (= 2 a 256)
LAYERS        = (1, 2)       # layer sommati nella mappa (None = tutti e 3): il layer 0 è rumoroso
NUM_WORKERS   = 8
SEED          = 1

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
OUTPUT_DIR    = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


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

    train_dataset = AugmentedTrainDataset(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # l'encoder (teacher) è congelato: l'optimizer aggiorna solo bottleneck e decoder
    optimizer = torch.optim.Adam(list(model.decoder.parameters()) + list(model.bn.parameters()),
                                 lr=LR, betas=BETAS)
    total_steps = int(TRAIN_EPOCHS * len(train_loader))
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = AMP and device.type == "cuda"
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training RD4AD single-bee | {TRAIN_EPOCHS} epoche = {total_steps} step "
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
                maps, scores, labels, masks = run_inference(ema_model, test_loader, device)
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
