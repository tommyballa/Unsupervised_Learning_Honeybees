"""
CFA (Coupled-hypersphere-based Feature Adaptation) - Anomaly detection su
singole api (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
               val: 322 con varroa + 1251 sane, per la best e le soglie F1;
               test_heldout: 957 con varroa + 5000 sane, valutato una volta;
               vedi make_val_split.py).
               Lo sweep qui sotto è stato fatto sul dataset batch3+4 (23k api):
               la configurazione è la stessa, con BANK_PATCHES_PER_IMAGE ridotto
               a 16 (stesso numero totale di patch per il KMeans, ~1.6M) e il
               training definito in step (TRAIN_STEPS) invece che in epoche.
Addestramento: backbone wide_resnet50_2 ImageNet congelato, memory bank
               costruita con le feature (descriptor iniziale) delle api sane,
               patch descriptor addestrato con la soft-boundary loss (AdamW,
               wd 5e-4, amsgrad). Score immagine = massimo della heatmap
               smussata (paper).
Ottimizzazione (sweep: experiments/single_bee/sweep_cfa.py; AUROC immagine,
i numeri "senza training" sono con il descriptor iniziale):
  - il CFA del paper (memory bank = media per posizione delle feature, 384 px,
    layer 1-2, lr 1e-3) si ferma a ~0.72: le api non sono allineate e l'86%
    del crop è sfondo nero, quindi la bank è quasi tutta sfondo;
  - memory bank = centroidi KMeans di patch campionate solo sull'ape
    (BANK_INIT "kmeans" + bee_foreground), loss e mappa solo sull'ape:
    0.80 senza training con 16k centroidi (0.73 con 4k); a 512 px 0.85;
  - K = 1 (distanza dal solo centroide più vicino): 0.87 contro 0.85 con
    K = 5, il softmin sui K vicini sporca lo score;
  - layer 2-3 al posto di 1-2: 0.94 con 32k centroidi (layer 1-3: 0.93;
    64k centroidi: +0.001, non vale il costo doppio);
  - crop ingranditi a 768 px (le varroe sono piccole e i layer 2-3 hanno
    stride 8): 0.966 (a 1024 px 0.965, con mappe peggiori);
  - training breve con lr basso: con 23k immagini il lr del paper (1e-3, pensato
    per ~500 step su ~250 immagini) fa collassare il descriptor, che schiaccia
    le feature sulla bank (da 0.80 a 0.62 in 360 step; con 1e-4 dopo ~90 step).
    Con lr 1e-5 l'adattamento porta da 0.966 a 0.973 con un picco tra 260 e
    430 step, poi cala: poche centinaia di step con LR coseno e valutazione
    frequente (step 0 = nessun adattamento).
Best:          ogni EVAL_INTERVAL step (e alla fine) metriche sul validation set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/cfa/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee cfa
"""

import math
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
from moviad.models.cfa.cfa import CFA, CFATrainArgs
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
MODEL_NAME    = "cfa"
REPO_ROOT     = Path(__file__).resolve().parents[2]
DATASET_DIR   = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR     = f"{DATASET_DIR}/train/normal"
VAL_DIR       = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR      = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

# layer a stride 8 e 16 e canali concatenati; backbone diversi solo per
# l'ablation (ABL_BACKBONE, vedi ablation.py)
BACKBONE_LAYERS = {
    "wide_resnet50_2": (["layer2", "layer3"], 1536),       # 512 + 1024
    "resnet18":        (["layer2", "layer3"], 384),        # 128 + 256
    "mobilenet_v2":    (["features.6", "features.13"], 128),  # 32 + 96
}
BACKBONE      = ablation.backbone("wide_resnet50_2")
LAYERS, FEATURE_CHANNELS = BACKBONE_LAYERS[BACKBONE]
# gamma_d=2: canali del patch descriptor dimezzati (1536 -> 768)
GAMMA_D       = 2
IMG_SIZE      = (768, 768)   # crop 512x512 ingranditi: mappa 96x96 (layer2, stride 8)
# memory bank: centroidi KMeans di BANK_PATCHES_PER_IMAGE patch casuali per
# immagine prese sull'ape ("mean" = media per posizione, come nel paper)
BANK_INIT     = "kmeans"
BANK_SIZE     = 32768
BANK_PATCHES_PER_IMAGE = 16   # 102k immagini -> ~1.6M patch (64 con le 23k di batch3+4)
if ablation.TRAIN_N:
    # meno api: stesse ~1.6M patch candidate al KMeans, al massimo 256 per ape
    BANK_PATCHES_PER_IMAGE = min(256, -(-BANK_PATCHES_PER_IMAGE * 102_303 // ablation.TRAIN_N))
K             = 1            # centroidi vicini usati da loss (attrazione) e score
J             = 3            # centroidi successivi usati dalla loss (repulsione)
# maschera dell'ape: pixel non neri (lo sfondo dei crop è nero, ~86% dell'immagine),
# chiusura per riempire le zone scure del corpo, poi margine attorno all'ape
# (in pixel a GT_SIZE, come PatchCore)
FG_THRESHOLD  = 10 / 255
FG_CLOSING    = 15
FG_DILATION   = 8
TRAIN_STEPS   = 522          # il picco è dopo poche centinaia di step (= 0.36 epoche di batch3+4)
BATCH_SIZE    = 16
TEST_BATCH_SIZE = 32
LR            = 1e-5         # paper: 1e-3 (qui fa collassare il descriptor)
WEIGHT_DECAY  = 5e-4
WARMUP_FRAC   = 0.02         # warmup lineare, poi decadimento coseno fino a 0
EVAL_INTERVAL = 87           # valuta ogni N step e sempre alla fine
BLUR_SIGMA    = 4.0          # sigma del blur sulla mappa a IMG_SIZE
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


def _dilate(mask, size):
    return F.max_pool2d(mask, size, stride=1, padding=size // 2)


def bee_foreground(images):
    """Maschera (B,1,H,W) dell'ape da immagini normalizzate ImageNet."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    rgb = images * std + mean
    mask = (rgb.amax(dim=1, keepdim=True) > FG_THRESHOLD).float()
    scale = images.shape[-1] / GT_SIZE[1]
    closing = 2 * round(FG_CLOSING * scale / 2) + 1
    mask = 1 - _dilate(1 - _dilate(mask, closing), closing)    # chiusura
    return _dilate(mask, 2 * round(FG_DILATION * scale) + 1)


def build_model(device):
    feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
    model = CFA(
        feature_extractor=feature_extractor,
        backbone=BACKBONE,
        gamma_d=GAMMA_D,
        feature_maps_channels=FEATURE_CHANNELS,
        K=K,
        J=J,
        bank_init=BANK_INIT,
        bank_patches_per_image=BANK_PATCHES_PER_IMAGE,
        bank_size=BANK_SIZE,
        blur_sigma=BLUR_SIGMA,
        foreground_fn=bee_foreground,
    )
    model.to(device)
    return model


def model_state(model):
    # patch descriptor, raggio r e memory bank (il backbone congelato non è nello state dict)
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    # in un modello nuovo la memory bank non esiste ancora (viene creata prima del training)
    model.memory_bank = torch.nn.Parameter(state["memory_bank"].clone().to(model.device),
                                           requires_grad=False)
    model.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_cfa_single_bee():
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
        val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # AdamW(wd 5e-4, amsgrad) su patch descriptor e raggio + soft-boundary loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    training_args = CFATrainArgs(epochs=math.ceil(TRAIN_STEPS / len(train_loader)), batch_size=BATCH_SIZE, optimizer=optimizer)
    training_args.init_train(model)
    total_steps = TRAIN_STEPS
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    print(f"\nMemory bank ({BANK_INIT}): feature delle api sane con il descriptor iniziale")
    model.train()
    model.memory_bank = torch.nn.Parameter(model.initialize_memory_bank(train_loader), requires_grad=False)

    print(f"\n{'='*60}")
    print(f"Training CFA single-bee | {total_steps} step = {total_steps / len(train_loader):.2f} epoche "
          f"| {len(train_loader)} batch/epoca | valutazione ogni {EVAL_INTERVAL} step")
    print(f"{'='*60}\n")

    def evaluate(step):
        maps, scores, labels, masks = run_inference(model, val_loader, device)
        results = compute_metrics(maps, scores, labels, masks, with_pro=False)
        del maps, masks
        print_metrics(results)
        tracker.update(step, results, lambda: model_state(model))

    # step 0: memory bank con il descriptor iniziale (nessun adattamento)
    print("Step 0 (descriptor iniziale)")
    evaluate(0)

    step, running_loss, n = 0, 0.0, 0
    while step < total_steps:
        for batch in tqdm(train_loader, desc=f"step {step}/{total_steps}", leave=False):
            running_loss += model.train_step(batch, training_args)
            scheduler.step()
            step += 1; n += 1

            if step % EVAL_INTERVAL == 0 or step == total_steps:
                print(f"Step {step}/{total_steps} (epoca {step / len(train_loader):.2f}) — "
                      f"loss: {running_loss / n:.6f}, lr: {scheduler.get_last_lr()[0]:.2e}")
                running_loss, n = 0.0, 0
                evaluate(step)
            if step >= total_steps:
                break

    print(f"\nBest CFA (\"epoca\" = step di training) — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_cfa_single_bee()
