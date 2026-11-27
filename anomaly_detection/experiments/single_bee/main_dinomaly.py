"""
Dinomaly training ottimizzato per rilevamento varroa su api singole.

Ottimizzazioni velocità:
  - torch.compile su encoder blocks (frozen) e decoder blocks
  - AMP bfloat16 su forward + loss
  - batch_size=32 (A6000 48GB lo regge comodamente)
  - Valutazione ogni EVAL_EVERY iterazioni
  - persistent_workers + pin_memory nei DataLoader
  - cudnn.benchmark=True (input size fisso)
  - zero_grad(set_to_none=True) + non_blocking transfers

Ottimizzazioni qualità varroa:
  - Data augmentation geometrica (flips, rotazione)
  - Maschere GT, metriche e heatmap a 512x512 (risoluzione dei crop)

Allineamento all'implementazione ufficiale di Dinomaly:
  - loss coseno GLOBALE con hard mining (components.global_cosine_hm_percent)
  - mappa smussata con gaussiano (kernel 5, sigma 4) e score immagine = media
    del top-1% dei pixel (Dinomaly(max_ratio=0.01))
  - valutazione in fp32 (il training resta in bf16)

Dataset: single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
val: 322 con varroa + 1251 sane, per la best e le soglie F1;
test_heldout: 957 con varroa + 5000 sane, valutato una volta;
vedi make_val_split.py).
Con 102k immagini un'epoca è ~3200 iterazioni: il training è a iterazioni
(TOTAL_ITERS = 10k come nel repo ufficiale, scheduler coseno su TOTAL_ITERS).
Su train_single_bee_merged (23k immagini) la configurazione ufficiale ViT-B
(layer 2-9 in due gruppi, ~7k iterazioni) aveva dato img AUROC 0.960 contro
0.949 dei layer 2-5 (pxl AP 0.574 contro 0.587).

Best: ogni EVAL_EVERY iterazioni (e all'ultima) metriche sul validation set; la
configurazione migliore (img_roc_auc, a parità pxl_avg_prec) viene salvata in
checkpoints/single_bee/dinomaly/ (best.pth = bottleneck + decoder, l'encoder
DINOv2 è congelato; config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee dinomaly
"""

from moviad.datasets.varroa.paths import DATA_ROOT
from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import (
    IMAGENET_MEAN, IMAGENET_STD, compute_metrics, print_metrics, run_inference,
)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import os
import random
from pathlib import Path
import numpy as np

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME     = "dinomaly"
# vit_small_patch14_reg4_dinov2 (stessi layer, come nel repo ufficiale): ablation (ABL_BACKBONE)
ENCODER_NAME   = ablation.backbone("vit_base_patch14_reg4_dinov2")
# configurazione ufficiale di Dinomaly per ViT-B: 8 layer intermedi fusi in due gruppi
TARGET_LAYERS  = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
MAX_RATIO      = 0.01   # score immagine = media del top-1% dei pixel (come nel repo ufficiale)
IMG_SIZE       = (518, 518)
TOTAL_ITERS    = 10_000  # iterazioni di training (~3 epoche)
BATCH_SIZE     = 32
EVAL_EVERY     = 1000   # valuta ogni N iterazioni (e sempre all'ultima)
SEED           = 1
NUM_WORKERS    = 8

REPO_ROOT      = Path(__file__).resolve().parents[2]
DATASET_DIR    = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR      = f"{DATASET_DIR}/train/normal"
VAL_DIR        = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR       = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE        = (512, 512)

NUM_HEATMAPS   = 20      # metà sane, metà con varroa
OUTPUT_DIR     = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
CHECKPOINT_DIR = Path(os.environ.get("DINOMALY_CKPT_DIR",
                                    REPO_ROOT / "checkpoints" / "single_bee_val" / MODEL_NAME))
if ablation.ACTIVE:
    CHECKPOINT_DIR, OUTPUT_DIR = ablation.dirs(REPO_ROOT, MODEL_NAME, CHECKPOINT_DIR, OUTPUT_DIR)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True è sicuro con input size fisso e velocizza le convoluzioni
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Dataset wrapper con data augmentation
# ──────────────────────────────────────────────────────────────────────
class AugmentedTrainDataset(Dataset):
    """
    Wrappa il VarroaDataset (che cacha i tensori trasformati) e applica
    augmentation geometrica random ad ogni accesso.

    L'augmentation opera su tensori già normalizzati: le trasformazioni
    geometriche (flip, rotazione) non alterano la distribuzione dei valori.
    Questo è fondamentale per il rilevamento varroa perché le api possono
    apparire in qualsiasi orientamento.
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15, fill=0),
            ], p=0.3),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image = self.base[idx]  # tensore cachato (C, H, W)
        return self.aug(image)


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


def build_model(device):
    model = Dinomaly(
        encoder_name=ENCODER_NAME,
        target_layers=TARGET_LAYERS,
        fuse_layer_encoder=FUSE_LAYER_ENCODER,
        fuse_layer_decoder=FUSE_LAYER_DECODER,
        # il decoder usa LinearAttention2, che ignora l'attention mask: il
        # neighbor masking non avrebbe effetto (e non è usato nel repo ufficiale)
        mask_neighbor_size=0,
        img_size=IMG_SIZE,
        max_ratio=MAX_RATIO,
    )
    return model.to(device)


def model_state(model):
    # encoder DINOv2 congelato: bastano bottleneck + decoder (senza i prefissi di torch.compile)
    return clean_state_dict(model.trainable.state_dict())


def load_model_state(model, state):
    model.trainable.load_state_dict(state)


# ──────────────────────────────────────────────────────────────────────
# Training diretto (bypass ContinualTrainer per massima velocità)
# ──────────────────────────────────────────────────────────────────────
def train_dinomaly():

    set_seed(SEED)
    torch.set_float32_matmul_precision('high')

    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Modello ──────────────────────────────────────────────────────
    model = build_model(device)

    # ── torch.compile sui blocchi computazionali ─────────────────────
    # L'encoder è congelato con input size fisso → compilazione perfetta
    print("Compiling encoder blocks...")
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            model.encoder.blocks[i] = torch.compile(blk)

    # Decoder e bottleneck: compilazione per accelerare il backward pass
    print("Compiling decoder and bottleneck blocks...")
    for i, blk in enumerate(model.decoder):
        model.decoder[i] = torch.compile(blk)
    for i, blk in enumerate(model.bottleneck):
        model.bottleneck[i] = torch.compile(blk)

    # ── Dataset ──────────────────────────────────────────────────────
    # Training dataset con augmentation geometrica
    train_dataset = AugmentedTrainDataset(ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN)))

    # Test dataset (senza augmentation)
    val_dataset = VarroaDataset(dataset_args(VAL_DIR), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Training args ────────────────────────────────────────────────
    training_args = DinomalyTrainArgs(
        epochs=-(-TOTAL_ITERS // len(train_loader)),
        batch_size=BATCH_SIZE,
        total_iters=TOTAL_ITERS,
    )
    training_args.init_train(model)
    # la "best_epoch" salvata dal tracker è il numero di iterazioni
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()), selection_data="validation set")

    # ── Training loop (a iterazioni) ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Dinomaly | {TOTAL_ITERS} iterazioni | {len(train_loader)} batch/epoca")
    print(f"Batch size: {BATCH_SIZE} | Eval ogni {EVAL_EVERY} iterazioni")
    print(f"{'='*60}\n")

    def batches():
        while True:
            yield from train_loader

    model.train()
    running_loss = 0.0
    for it, batch in enumerate(tqdm(batches(), total=TOTAL_ITERS), start=1):
        running_loss += model.train_step(batch, training_args)

        # Valutazione (ogni EVAL_EVERY iterazioni e all'ultima) + salvataggio della best
        if it % EVAL_EVERY == 0 or it == TOTAL_ITERS:
            n = EVAL_EVERY if it % EVAL_EVERY == 0 else it % EVAL_EVERY
            print(f"\nITER: {it} | Loss: {running_loss / n:.6f}")
            running_loss = 0.0
            maps, scores, labels, masks = run_inference(model, eval_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(it, results, lambda: model_state(model))
            del maps, scores, labels, masks
            model.train()

        if it == TOTAL_ITERS:
            break

    print(f"\nBest Dinomaly — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_dinomaly()
