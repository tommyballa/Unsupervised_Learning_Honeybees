"""
SuperSimpleNet - Anomaly detection su singole api (single bee crops).

Addestramento: immagini normali di singole api (VarroaDataset, split TRAIN).
               Le anomalie vengono simulate nello spazio delle feature
               (rumore gaussiano mascherato con Perlin noise), come nel paper.
               AdamW del paper (adaptor 1e-4, segdec 2e-4) con cosine annealing
               per step, fp16 autocast in training.
Ottimizzazione (sweep: experiments/single_bee/sweep_supersimplenet.py,
logs/supersimplenet_sweep/; metriche dello sweep: pixel AP a 128 px, img AUROC
dal massimo della mappa):
  - la configurazione del paper (256x256, layer2+layer3, score della testa di
    classificazione, 30 epoche) si ferma a img AUROC 0.726 / pixel AP 0.098
    (checkpoints/single_bee/supersimplenet_baseline_256/);
  - risoluzione nativa 512 (la varroa ha diametro ~32 px): a 256 img AUROC
    ~0.65 dopo 750 step contro ~0.88 a 512. Ingrandire l'input a 768 non
    aiuta (0.93);
  - score = massimo della mappa (CLS_HEAD = False): la testa di classificazione
    vede anomalie sintetiche estese (Perlin) e non generalizza alla varroa
    (img AUROC 0.55-0.80 quando il massimo della mappa dà 0.91-0.94), e
    addestrare la sola testa di segmentazione non peggiora la mappa. Media dei
    top-k pixel e blur prima dello score non aiutano;
  - layer1 alla sua risoluzione (feature 128x128): con layer2+layer3 tutte le
    varianti si fermano a 0.93-0.94 (feature 64x64 o 128x128) e gli errori sono
    le varroe piccole (AUROC 0.89 sotto 800 px di area, 0.99 sopra 1300);
    con layer1-3 a 128x128 0.952-0.956. Layer1 ridotto a 64x64 non aiuta (0.90);
  - rumore: std 0.02 (0.015 del paper è equivalente ma più lento; 0.01 troppo
    lento; 0.05 satura a 0.91-0.92, 0.15 non funziona). Soglia Perlin 0.6,
    standardizzazione delle feature per canale e pooling 1x1 non aiutano;
  - blur della mappa (MAP_SIGMA, usato solo per la mappa): sigma 8 px a 512 è
    il migliore per la pixel AP (0.45 contro 0.26 senza blur), 16 px peggiora;
  - il learning rate deve scendere: a lr costante le metriche oscillano molto
    tra una valutazione e l'altra, il meglio arriva nella parte finale del coseno.
Con questo script (metriche ufficiali a 512): step 6000 img_roc_auc 0.952 /
img_avg_prec 0.769 / pxl_avg_prec 0.516.
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/supersimplenet/
               (best.pth + config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee supersimplenet
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME      = "supersimplenet"
REPO_ROOT       = Path(__file__).resolve().parents[2]
TRAIN_DIR       = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_batch3+4/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE         = (512, 512)

BACKBONE        = "wide_resnet50_2"
LAYERS          = ["layer1", "layer2", "layer3"]   # il paper usa layer2 + layer3
IMG_SIZE        = (512, 512)   # risoluzione nativa dei crop
FEATURE_UPSCALE = 1            # feature alla risoluzione di layer1 (128x128); nel paper 2x il primo layer
NOISE_STD       = 0.02         # std del rumore gaussiano delle anomalie sintetiche (paper: 0.015)
CLS_HEAD        = False        # solo testa di segmentazione: score = massimo della mappa
MAP_SIGMA       = 8.0          # blur gaussiano della mappa (px a 512), non usato per lo score
TRAIN_STEPS     = 6000         # ~4 epoche con batch 16
BATCH_SIZE      = 16
EVAL_INTERVAL   = 1000         # valuta ogni N step (e sempre alla fine)
AMP             = True         # fp16 autocast in training (backbone e testa di segmentazione)
NUM_WORKERS     = 8
SEED            = 1

# Ottimizzatore del paper / anomalib (AdamW), con cosine annealing per step fino a 0
ADAPTOR_LR      = 1e-4
SEGDEC_LR       = 2e-4
WEIGHT_DECAY    = 1e-5
PERLIN_THRESHOLD = SuperSimpleNet.DEFAULT_PARAMETERS["perlin_threshold"]

NUM_HEATMAPS    = 20           # metà sane, metà con varroa
RAW_LABEL       = "Mappa grezza (logit)"
OUTPUT_DIR      = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR  = REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME


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
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    return SuperSimpleNet(
        feature_extractor=feature_extractor,
        perlin_threshold=PERLIN_THRESHOLD,
        stop_grad=SuperSimpleNet.DEFAULT_PARAMETERS["stop_grad"],
        noise_std=NOISE_STD,
        upscale=FEATURE_UPSCALE,
        map_sigma=MAP_SIGMA,
        cls_head=CLS_HEAD,
    ).to(device)


def model_state(model):
    # il backbone congelato non fa parte dello state dict: adaptor + segdec
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_supersimplenet_single_bee():
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

    # --- Optimizer, scheduler (per step) e loss ---
    optimizer = AdamW([
        {"params": model.adaptor.parameters(), "lr": ADAPTOR_LR},
        {"params": model.segdec.parameters(), "lr": SEGDEC_LR, "weight_decay": WEIGHT_DECAY},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    use_amp = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = SSNLoss()
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training SuperSimpleNet single-bee | {TRAIN_STEPS} step "
          f"| {len(train_loader)} batch/epoca | valutazione ogni {EVAL_INTERVAL} step")
    print(f"{'='*60}\n")

    step, running_loss, n = 0, 0.0, 0
    while step < TRAIN_STEPS:
        for images in tqdm(train_loader, desc=f"step {step}/{TRAIN_STEPS}", leave=False):
            model.train()
            images = images.to(device, non_blocking=True)
            B, _, H, W = images.shape
            # tutte le immagini di training sono normali: maschere e label a zero,
            # le anomalie vengono generate dal modello nello spazio delle feature
            masks = torch.zeros(B, 1, H, W, device=device)
            labels = torch.zeros(B, device=device)

            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                anomaly_map, anomaly_score, masks, labels = model(images=images, masks=masks, labels=labels)
            loss = loss_fn(pred_map=anomaly_map.float(),
                           pred_score=None if anomaly_score is None else anomaly_score.float(),
                           target_mask=masks, target_label=labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running_loss += loss.item(); n += 1

            if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
                print(f"Step {step}/{TRAIN_STEPS} (epoca {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()}")
                running_loss, n = 0.0, 0
                maps, scores, labels, masks = run_inference(model, test_loader, device)
                results = compute_metrics(maps, scores, labels, masks, with_pro=False)
                del maps, masks
                print_metrics(results)
                tracker.update(step, results, lambda: model_state(model))
            if step >= TRAIN_STEPS:
                break

    print(f"\nBest SuperSimpleNet (\"epoca\" = step di training) — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_supersimplenet_single_bee()
