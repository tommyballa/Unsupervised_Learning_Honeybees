"""
PaDiM - Anomaly detection su singole api (single bee crops).

Dataset:       single_bee3+4+5+6_SAM3_varroa_split (train/normal: 102k api sane;
               val: 322 con varroa + 1251 sane, per la best e le soglie F1;
               test_heldout: 957 con varroa + 5000 sane, valutato una volta;
               vedi make_val_split.py).
Addestramento: feature di un backbone ImageNet congelato (layer concatenati alla
               risoluzione del primo, riduzione casuale a PROJ_DIM canali come nel
               paper) delle api sane (VarroaDataset, TRAIN), stima di una
               gaussiana multivariata (covarianza + COV_EPS I, 0.01 nel paper).
               Media e covarianza sono accumulate in streaming su GPU (somme e
               prodotti esterni in float64), quindi si usano tutte le immagini
               di training senza tenerne le feature in RAM.
               GAUSSIAN = "position": una gaussiana per posizione della griglia
               (PaDiM originale); "shared": una sola gaussiana per tutte le patch
               dell'ape (le api non sono allineate, la posizione non è informativa).
               Nessuna epoca da scegliere: media e covarianza vengono
               salvate in checkpoints/single_bee/padim/ (best.pth + config.json).
Valutazione:   distanza di Mahalanobis per patch, mappa riportata a GT_SIZE,
               blur gaussiano MAP_SIGMA px (a GT_SIZE), azzerata fuori dall'ape; score
               immagine = massimo della mappa.
Ottimizzazione (sweep: experiments/single_bee/sweep_padim.py, dataset finale,
20k immagini di training, AUROC immagine, AP pixel a 128 px; logs/padim_sweep/):
  - PaDiM originale (wide_resnet50_2 layer1-3, 550 canali casuali, gaussiana per
    posizione, eps 0.01, 256 px): 0.65 / 0.02. La gaussiana per posizione
    peggiora sempre (le api non sono allineate): con layer2-3 a 512 px 0.78
    contro 0.89 della gaussiana condivisa;
  - eps 0.01 schiaccia la Mahalanobis verso una distanza euclidea: 1e-4 è il
    migliore quasi ovunque (layer2-3 512 px condivisa, tutti i canali: 0.937 -> 0.962);
  - 512 px invece di 256: 0.70 -> 0.89; 768 px peggiora (0.96);
  - tutti i canali invece della random projection a 550: 0.89 -> 0.962;
  - il layer1 rovina tutto (layer1-2: 0.54, layer2 da solo 0.59); solo
    layer3: 0.983 / 0.56; resnet18 layer2-3: 0.94;
  - avg pool 3x3 delle feature: layer2-3 0.985, layer3 0.989 / 0.565 (scelto:
    metà dei canali e metà del tempo di layer2-3 con pool 5, 0.989 / 0.562);
  - blur della mappa: sigma 0-4 equivalenti (0.989 / 0.988), 16 peggiora.
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO):
    python experiments/plot_heatmaps.py single_bee padim
"""

from moviad.datasets.varroa.paths import DATA_ROOT
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import save_checkpoint, script_config
from moviad.utilities.evaluation.report import IMAGENET_MEAN, IMAGENET_STD

import ablation


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "padim"
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATASET_DIR  = DATA_ROOT + "/processed/single_bee3+4+5+6_SAM3_varroa_split"
TRAIN_DIR    = f"{DATASET_DIR}/train/normal"
VAL_DIR      = f"{DATASET_DIR}/val"           # selezione della best e soglie F1
TEST_DIR     = f"{DATASET_DIR}/test_heldout"  # valutato una sola volta sulla best
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE      = (512, 512)

# layer a stride 16; backbone diversi solo per l'ablation (ABL_BACKBONE, vedi ablation.py)
BACKBONE_LAYERS = {
    "wide_resnet50_2": ["layer3"],
    "resnet18":        ["layer3"],
    "mobilenet_v2":    ["features.13"],
}
BACKBONE     = ablation.backbone("wide_resnet50_2")
LAYERS       = BACKBONE_LAYERS[BACKBONE]
PROJ_DIM     = 0            # canali tenuti dalla random projection (0 = tutti)
GAUSSIAN     = "shared"     # "position" (paper) | "shared"
FEAT_POOL    = 3            # avg pool locale delle feature (1 = nessuno)
COV_EPS      = 1e-4         # 0.01 nel paper: troppo per feature con varianza ~0.02
MAP_SIGMA    = 4            # px a GT_SIZE, blur gaussiano della mappa
# maschera dell'ape: pixel non neri (lo sfondo dei crop è nero, ~86% dell'immagine),
# chiusura per riempire le zone scure del corpo, poi margine attorno all'ape
FG_THRESHOLD = 10 / 255
FG_CLOSING   = 15           # px (lato del kernel)
FG_DILATION  = 8            # px di margine
IMG_SIZE     = (512, 512)
TRAIN_IMAGES = 0            # sottoinsieme casuale del training set (0 = tutte)
BATCH_SIZE   = 32
TEST_BATCH_SIZE = 32
NUM_WORKERS  = 8
SEED         = 1

NUM_HEATMAPS = 20           # metà sane, metà con varroa
OUTPUT_DIR   = REPO_ROOT / "heatmaps" / "single_bee_val" / MODEL_NAME
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
    mask = 1 - _dilate(1 - _dilate(mask, FG_CLOSING), FG_CLOSING)    # chiusura
    return _dilate(mask, 2 * FG_DILATION + 1)


class PadimSingleBee(torch.nn.Module):
    """PaDiM con stima in streaming e gaussiana per posizione o condivisa."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.feature_extractor = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
        with torch.no_grad():
            feats = self.feature_extractor(torch.zeros(1, 3, *IMG_SIZE, device=device))
        total_dim = sum(f.shape[1] for f in feats)
        self.grid = tuple(feats[0].shape[-2:])
        g = torch.Generator().manual_seed(SEED)
        dims = torch.randperm(total_dim, generator=g)[:PROJ_DIM] if PROJ_DIM else torch.arange(total_dim)
        self.register_buffer("dims", dims.sort().values.to(device))
        self.mean = None            # (C,) o (C, HW)
        self.cov = None             # (C, C) o (HW, C, C), senza regolarizzazione
        self.precision = None       # inversa di cov + COV_EPS I

    @torch.no_grad()
    def embed(self, images):
        """Embedding (B, C, h, w) alla risoluzione del primo layer."""
        feats = self.feature_extractor(images)
        feats = [f if f.shape[-2:] == self.grid else
                 F.interpolate(f, size=self.grid, mode="nearest") for f in feats]
        emb = torch.cat(feats, 1).index_select(1, self.dims)
        if FEAT_POOL > 1:
            emb = F.avg_pool2d(emb, FEAT_POOL, stride=1, padding=FEAT_POOL // 2)
        return emb.float()

    def grid_foreground(self, images):
        """Celle della griglia coperte per almeno metà dall'ape (B, h*w)."""
        fg = F.adaptive_avg_pool2d(bee_foreground(images), self.grid)
        return (fg >= 0.5).flatten(1)

    @torch.no_grad()
    def fit(self, loader):
        n, s1, s2, shift = 0, None, None, None
        for batch in tqdm(loader, desc="Fitting PaDiM"):
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
            emb = self.embed(images).flatten(2).double()            # (B, C, HW)
            if shift is None:
                shift = emb.mean(0) if GAUSSIAN == "position" else emb.mean((0, 2))
            if GAUSSIAN == "position":
                x = (emb - shift).permute(2, 0, 1)                    # (HW, B, C)
                if s1 is None:
                    s1 = torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
                    s2 = torch.zeros(x.shape[0], x.shape[2], x.shape[2], dtype=x.dtype, device=x.device)
                s1 += x.sum(1)
                s2.baddbmm_(x.transpose(1, 2), x)                     # in place: s2 è grande
                n += x.shape[1]
            else:
                fg = self.grid_foreground(images)
                x = (emb - shift[None, :, None]).permute(0, 2, 1)[fg]  # (N_fg, C)
                s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
                s2 = x.T @ x if s2 is None else s2 + x.T @ x
                n += x.shape[0]
        self.set_statistics(n, s1, s2, shift)

    def set_statistics(self, n, s1, s2, shift):
        mu = s1 / n                                                  # (HW, C) o (C,)
        s2 -= n * mu[..., :, None] * mu[..., None, :]                # in place: s2 è grande
        s2 /= n - 1
        self.cov = s2.float()
        if GAUSSIAN == "position":
            self.mean = (mu + shift.T).T.float().contiguous()          # (C, HW)
        else:
            self.mean = (mu + shift).float()
        self.set_precision(COV_EPS)

    def set_precision(self, eps):
        cov = self.cov.view(-1, *self.cov.shape[-2:])
        eye = torch.eye(cov.shape[-1], dtype=torch.float64, device=cov.device)
        precision = torch.empty_like(cov)
        for i in range(0, cov.shape[0], 256):                        # a blocchi di posizioni
            c = cov[i:i + 256].double() + eps * eye
            precision[i:i + 256] = torch.cholesky_inverse(torch.linalg.cholesky(c)).float()
        self.precision = precision.view(self.cov.shape)

    @torch.no_grad()
    def distances(self, emb):
        """Distanza di Mahalanobis per patch (B, 1, h, w)."""
        b, c, h, w = emb.shape
        if GAUSSIAN == "position":
            diff = (emb.flatten(2) - self.mean[None]).permute(2, 0, 1)    # (HW, B, C)
            d2 = ((diff @ self.precision) * diff).sum(-1).T                # (B, HW)
        else:
            diff = emb.flatten(2).transpose(1, 2) - self.mean              # (B, HW, C)
            d2 = ((diff @ self.precision) * diff).sum(-1)
        return d2.clamp_min(0).sqrt().reshape(b, 1, h, w)

    @torch.no_grad()
    def forward(self, images):
        dist = self.distances(self.embed(images))
        amap = F.interpolate(dist, size=GT_SIZE, mode="bilinear", align_corners=False)
        if MAP_SIGMA > 0:
            amap = gaussian_blur(amap, kernel_size=2 * int(4.0 * MAP_SIGMA + 0.5) + 1, sigma=MAP_SIGMA)
        amap = amap * bee_foreground(F.interpolate(images, size=GT_SIZE, mode="bilinear", align_corners=False))
        return amap, amap.flatten(1).amax(1)


def build_model(device):
    return PadimSingleBee(device)


def model_state(model):
    return {"mean": model.mean.cpu(), "cov": model.cov.cpu(), "dims": model.dims.cpu()}


def load_model_state(model, state):
    model.mean = state["mean"].to(model.device)
    model.cov = state["cov"].to(model.device)
    model.dims = state["dims"].to(model.device)
    model.set_precision(COV_EPS)


def train_dataset():
    dataset = ablation.subset_train(VarroaDataset(dataset_args(), "varroa", Split.TRAIN))
    if TRAIN_IMAGES and TRAIN_IMAGES < len(dataset):
        subset = np.random.default_rng(SEED).choice(len(dataset), TRAIN_IMAGES, replace=False)
        dataset = torch.utils.data.Subset(dataset, sorted(subset.tolist()))
    return dataset


# ──────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────
def train_padim_single_bee():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(device)
    dataset = train_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\nFitting PaDiM ({BACKBONE}, {LAYERS}, {len(model.dims)} canali, gaussiana {GAUSSIAN}, "
          f"griglia {model.grid}) su {len(dataset)} api sane...")
    model.fit(loader)

    save_checkpoint(CHECKPOINT_DIR, model_state(model), script_config(globals()),
                    best_epoch=None, note="fitting in un solo passaggio (nessuna epoca da scegliere)")
    print(f"Gaussiane PaDiM salvate in {CHECKPOINT_DIR}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_padim_single_bee()
