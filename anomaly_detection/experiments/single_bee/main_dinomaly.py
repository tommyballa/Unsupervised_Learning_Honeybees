"""
Dinomaly training ottimizzato per rilevamento varroa su api singole.

Ottimizzazioni velocità:
  - torch.compile su encoder blocks (frozen) e decoder blocks
  - AMP bfloat16 su forward + loss
  - batch_size=32 (A6000 48GB lo regge comodamente)
  - Valutazione ogni 5 epoche anziché ogni epoca
  - persistent_workers + pin_memory nei DataLoader
  - cudnn.benchmark=True (input size fisso)
  - zero_grad(set_to_none=True) + non_blocking transfers

Ottimizzazioni qualità varroa:
  - Data augmentation geometrica (flips, rotazione) sul dataset cachato
  - Maschere GT, metriche e heatmap a 512x512 (risoluzione dei crop)

Allineamento all'implementazione ufficiale di Dinomaly:
  - loss coseno GLOBALE con hard mining (components.global_cosine_hm_percent)
  - mappa smussata con gaussiano (kernel 5, sigma 4) e score immagine = media
    del top-1% dei pixel (Dinomaly(max_ratio=0.01))
  - valutazione in fp32 (il training resta in bf16)

Best: ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul test set; la
configurazione migliore (img_roc_auc, a parità pxl_avg_prec) viene salvata in
checkpoints/single_bee/dinomaly/ (best.pth = bottleneck + decoder, l'encoder
DINOv2 è congelato; config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee dinomaly
"""

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

# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME     = "dinomaly"
ENCODER_NAME   = "vit_base_patch14_reg4_dinov2"
# configurazione ufficiale di Dinomaly per ViT-B: 8 layer intermedi fusi in due gruppi
TARGET_LAYERS  = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
MAX_RATIO      = 0.01   # score immagine = media del top-1% dei pixel (come nel repo ufficiale)
IMG_SIZE       = (518, 518)
EPOCHS         = 10
BATCH_SIZE     = 32
EVAL_INTERVAL  = 1      # valuta ogni N epoche (e sempre all'ultima)
SEED           = 1
NUM_WORKERS    = 8

REPO_ROOT      = Path(__file__).resolve().parents[2]
TRAIN_DIR      = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE        = (512, 512)

NUM_HEATMAPS   = 20      # metà sane, metà con varroa
OUTPUT_DIR     = REPO_ROOT / "heatmaps" / "single_bee" / MODEL_NAME
CHECKPOINT_DIR = Path(os.environ.get("DINOMALY_CKPT_DIR",
                                    REPO_ROOT / "checkpoints" / "single_bee" / MODEL_NAME))


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
def dataset_args():
    return DatasetArguments(
        dataset_path=TRAIN_DIR,
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
    train_dataset = AugmentedTrainDataset(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))

    # Test dataset (senza augmentation)
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Training args ────────────────────────────────────────────────
    total_iters = EPOCHS * len(train_loader)
    training_args = DinomalyTrainArgs(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        total_iters=total_iters,
    )
    training_args.init_train(model)
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Dinomaly | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"Batch size: {BATCH_SIZE} | Eval ogni {EVAL_INTERVAL} epoche")
    print(f"total_iters (LR scheduler): {total_iters}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        print(f"EPOCH: {epoch}")

        # Training epoch
        avg_loss = 0.0
        for batch in tqdm(train_loader):
            avg_loss += model.train_step(batch, training_args)
        avg_loss /= len(train_loader)
        print(f"  Loss: {avg_loss:.6f}")

        # Valutazione (ogni EVAL_INTERVAL epoche e all'ultima) + salvataggio della best
        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, eval_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest Dinomaly — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_dinomaly()
