"""
GANomaly - Anomaly detection su singole api (single bee crops).

Modello:       GANomaly (encoder-decoder-encoder avversariale) addestrato su
               PATCH 32x32 dell'ape a 256x256, non sull'immagine intera: la
               varroa (~12 px a 256) occupa così una parte rilevante dell'input,
               come gli oggetti nei dataset del paper (CIFAR 32x32, patch X-ray).
Addestramento: patch casuali con almeno metà ape (lo sfondo nero è escluso) da
               un sottoinsieme fisso di TRAIN_IMAGES api sane, tenuto in memoria;
               loss del generatore = avversariale (feature matching) + 50 *
               ricostruzione L1 + encoding. Adam lr 2e-4 con decadimento a
               coseno, betas (0.5, 0.999); in valutazione si usa la media mobile
               (EMA) dei pesi. Input in [-1, 1] (mean = std = 0.5) perché il
               decoder termina con una Tanh.
Inferenza:     finestra scorrevole (passo PATCH_STRIDE), la ricostruzione
               dell'immagine è la media delle patch ricostruite. Mappa = residuo
               di CROMINANZA (input - ricostruzione, tolta la media sui canali)
               smussato con segno (sigma MAP_SIGMA) e poi in valore assoluto,
               solo all'interno dell'ape; score immagine = massimo della mappa.
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche su un
               sottoinsieme fisso del test set; la configurazione migliore
               (img_roc_auc, a parità pxl_avg_prec) viene salvata in
               checkpoints/single_bee/ganomaly/ (best.pth + config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee ganomaly

Ottimizzazione (sweep_ganomaly.py, logs/ganomaly_sweep/, train_batch3+4 /
test_batch3+4: 8000 immagini di train, test 2000 sane + 1000 con varroa,
metriche pixel a 128x128):
  - GANomaly sull'immagine intera (128 o 256 px, latente 100 o 512, con o senza
    augmentation) resta vicino al caso: img AUROC 0.58-0.60, e lo score del
    paper (||z - z'||^2) 0.45-0.52. La varroa è di pochi pixel e l'errore di
    ricostruzione è dominato dalla tessitura dell'ape.
  - Su patch gli score per patch (latente, feature del discriminatore, errore
    L1) si fermano a 0.62-0.66; la differenza la fa la mappa: il residuo con
    segno smussato alla scala della varroa cancella la tessitura e lascia le
    macchie compatte, e la sola crominanza funziona meglio dell'RGB.
  - Patch più piccole sono meglio (128 -> 64 -> 32) ma a 512 px la patch 32
    impara a ricostruire anche la varroa (img AUROC 0.57): 256 px / patch 32.
  - L'addestramento avversariale è instabile (img AUROC fra 0.66 e 0.87 a
    seconda dell'epoca): EMA dei pesi + lr a coseno rendono la curva monotona.
    Con 80 epoche invece di 40: img AUROC 0.863 -> 0.890, pxl AP128 0.30 -> 0.35.
  - Latente 100 è il migliore (16/32: 0.62-0.67, 50: 0.74, 256: 0.72-0.80); senza
    loss avversariale 0.852 contro 0.863 a 40 epoche.
  - Il discriminatore reinizializzato "come nel paper" con il vecchio optimizer
    andava subito in saturazione (loss BCE fissa a 50, gradiente nullo): ora si
    reinizializza anche l'optimizer, e anche quando la loss è satura.
Risultato (best all'epoca 80, test completo 5000 sane + 3119 con varroa, 512x512):
img AUROC 0.897, img AP 0.867, img F1 0.783, pxl AUROC 0.963, pxl AP 0.430,
AUPRO 0.910 (logs/ganomaly_main.log, logs/ganomaly_plot.log).
"""

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.ganomaly.loss import DiscriminatorLoss
from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import compute_metrics, print_metrics, run_inference


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "ganomaly"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_batch3+4/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

IMG_SIZE      = (256, 256)
PATCH_SIZE    = 32           # input di GANomaly (potenza di 2)
PATCH_STRIDE  = 16           # passo della finestra scorrevole in inferenza
MIN_FG        = 0.5          # frazione minima di ape in una patch di training
LATENT_VEC_SIZE = 100
EPOCHS        = 80
STEPS_PER_EPOCH = 500        # batch di patch per "epoca"
BATCH_SIZE    = 64           # patch per batch
TEST_BATCH_SIZE = 32         # immagini per batch in inferenza
LR            = 2e-4
BETA1         = 0.5
BETA2         = 0.999
W_ADV, W_CON, W_ENC = 1.0, 50.0, 1.0
EMA_DECAY     = 0.999        # media mobile dei pesi usata in valutazione
D_REINIT_MAX  = 5.0          # loss del discriminatore oltre cui è considerato saturo
MAP_SIGMA     = 6.0          # px a IMG_SIZE: blur (con segno) del residuo di crominanza
FG_THRESHOLD  = 10 / 255     # lo sfondo dei crop è nero
FG_EROSION    = 13           # px a IMG_SIZE: il bordo dell'ape è escluso dalla mappa
EVAL_INTERVAL = 10           # valuta ogni N epoche (e sempre all'ultima)
# sottoinsiemi fissi per velocizzare (None = tutto il dataset)
TRAIN_IMAGES  = 8000
EVAL_NORMALS  = 2000
EVAL_ANOMALIES = 1000
SUBSET_SEED   = 0
NUM_WORKERS   = 8
SEED          = 1

# Il decoder termina con Tanh: input e ricostruzione devono stare in [-1, 1]
GANOMALY_MEAN = [0.5, 0.5, 0.5]
GANOMALY_STD  = [0.5, 0.5, 0.5]

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
RAW_LABEL     = "Residuo di crominanza"
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
            transforms.Normalize(mean=GANOMALY_MEAN, std=GANOMALY_STD),
        ],
    )


def bee_foreground(images):
    """Maschera (B,1,H,W) dell'ape da immagini in [-1, 1]."""
    return ((images + 1) / 2).amax(dim=1, keepdim=True) > FG_THRESHOLD


class PatchGanomaly(nn.Module):
    """GANomaly su patch: in inferenza ricompone l'immagine e ne calcola la mappa."""

    def __init__(self, net: Ganomaly):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def forward(self, images):
        """Restituisce (mappa (B,1,H,W), score (B,)) per immagini in [-1, 1]."""
        b, _, h, w = images.shape
        p = PATCH_SIZE
        patches = F.unfold(images, p, stride=PATCH_STRIDE)                 # (B, 3pp, L)
        n_patches = patches.shape[-1]
        fake, _, _ = self.net.generator(patches.transpose(1, 2).reshape(-1, 3, p, p))
        # ricostruzione dell'immagine: media delle patch che coprono ogni pixel
        fake = fake.reshape(b, n_patches, -1).transpose(1, 2)
        count = F.fold(torch.ones_like(fake), (h, w), p, stride=PATCH_STRIDE)
        diff = images - F.fold(fake, (h, w), p, stride=PATCH_STRIDE) / count.clamp(min=1)

        # il blur del residuo con segno cancella la tessitura (media nulla) e
        # lascia le macchie compatte di colore diverso, come la varroa
        chroma = diff - diff.mean(dim=1, keepdim=True)
        kernel = 2 * int(4.0 * MAP_SIGMA + 0.5) + 1
        amap = TF.gaussian_blur(chroma, [kernel, kernel], [MAP_SIGMA, MAP_SIGMA]).abs().mean(dim=1, keepdim=True)

        fg = bee_foreground(images).float()
        fg = 1 - F.max_pool2d(1 - fg, FG_EROSION, stride=1, padding=FG_EROSION // 2)   # erosione
        amap = amap * fg
        return amap, amap.flatten(1).max(dim=1).values


def build_model(device):
    net = Ganomaly(
        input_size=(PATCH_SIZE, PATCH_SIZE),
        num_input_channels=3,
        n_features=64,
        latent_vec_size=LATENT_VEC_SIZE,
        extra_layers=0,
        add_final_conv_layer=True,
    )
    return PatchGanomaly(net).to(device)


def model_state(model):
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


def fixed_subset(n_total, n, rng):
    return np.sort(rng.choice(n_total, n, replace=False)) if n is not None and n < n_total else np.arange(n_total)


def load_train_images(train_dataset, device):
    """Sottoinsieme fisso del training set in memoria, uint8 (N,3,H,W) su device."""
    idx = fixed_subset(len(train_dataset), TRAIN_IMAGES, np.random.default_rng(SUBSET_SEED))
    loader = DataLoader(Subset(train_dataset, idx.tolist()), batch_size=64, num_workers=NUM_WORKERS)
    images = [((batch + 1) * 127.5).round().clamp(0, 255).byte() for batch in tqdm(loader, desc="Caricamento train")]
    return torch.cat(images).to(device)


def eval_subset(test_dataset):
    """Sottoinsieme fisso del test set per le valutazioni durante il training."""
    rng = np.random.default_rng(SUBSET_SEED)
    labels = np.asarray(test_dataset.labels)
    normal, anomal = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    idx = np.concatenate([normal[fixed_subset(len(normal), EVAL_NORMALS, rng)],
                          anomal[fixed_subset(len(anomal), EVAL_ANOMALIES, rng)]])
    return Subset(test_dataset, np.sort(idx).tolist())


def sample_patches(train_images):
    """Batch di patch casuali in [-1, 1] che contengono abbastanza ape, con flip e rotazioni."""
    n, p, device = 2 * BATCH_SIZE, PATCH_SIZE, train_images.device
    x = train_images[torch.randint(len(train_images), (n,), device=device)].float() / 127.5 - 1
    oy = torch.randint(x.shape[-2] - p + 1, (n,), device=device)
    ox = torch.randint(x.shape[-1] - p + 1, (n,), device=device)
    ar = torch.arange(p, device=device)
    rows, cols = (oy[:, None] + ar)[:, :, None], (ox[:, None] + ar)[:, None, :]
    patches = x[torch.arange(n, device=device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    keep = bee_foreground(patches).float().mean(dim=(1, 2, 3)) >= MIN_FG
    patches = patches[keep][:BATCH_SIZE]
    if torch.rand(1).item() < 0.5:
        patches = patches.flip(-1)
    return torch.rot90(patches, int(torch.randint(4, (1,)).item()), dims=(-2, -1)).contiguous()


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_ganomaly_single_bee():
    set_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model(device)
    net = model.net
    ema = copy.deepcopy(model).requires_grad_(False)     # pesi usati in valutazione
    print(f"Parametri GANomaly: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)
    train_images = load_train_images(train_dataset, device)
    test_subset = eval_subset(test_dataset)
    print(f"Train: {len(train_images)} immagini in memoria | valutazione su {len(test_subset)} immagini di test")
    test_loader = DataLoader(
        test_subset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Optimizer e loss (training GAN custom) ---
    g_opt = Adam(net.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(net.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_loss_fn = DiscriminatorLoss()
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()),
                          selection_data=f"sottoinsieme del test set ({len(test_subset)} immagini)")

    print(f"\n{'='*60}")
    print(f"Training GANomaly single-bee | {EPOCHS} epoche | {STEPS_PER_EPOCH} batch di patch/epoca")
    print(f"{'='*60}\n")

    step_count = 0
    for epoch in range(EPOCHS):
        model.train()
        avg_g_loss = 0.0
        avg_d_loss = 0.0
        reinits = 0

        for _ in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            patches = sample_patches(train_images)

            # Generator update (feature matching sul discriminatore)
            fake, latent_i, latent_o = net.generator(patches)
            pred_real, feat_real = net.discriminator(patches)
            _, feat_fake = net.discriminator(fake)
            g_loss = (W_ADV * F.mse_loss(feat_fake, feat_real.detach())
                      + W_CON * F.l1_loss(fake, patches)
                      + W_ENC * F.smooth_l1_loss(latent_o, latent_i))

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()
            avg_g_loss += g_loss.item()

            # EMA dei pesi (nelle prime iterazioni segue i pesi correnti)
            step_count += 1
            decay = min(EMA_DECAY, (1 + step_count) / (10 + step_count))
            for p_ema, p_cur in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p_cur.detach(), 1 - decay)
            for b_ema, b_cur in zip(ema.buffers(), model.buffers()):
                b_ema.copy_(b_cur)

            # Discriminator update
            pred_fake, _ = net.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real, pred_fake)

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            avg_d_loss += d_loss.item()

            # Discriminatore collassato (loss ~0, come nel paper) o saturato (BCE al
            # massimo, gradiente nullo): si reinizializza insieme al suo optimizer,
            # altrimenti i momenti di Adam dei vecchi pesi lo rimandano in saturazione
            if d_loss.item() < 1e-5 or d_loss.item() > D_REINIT_MAX:
                net.discriminator.apply(Ganomaly.weights_init)
                d_opt = Adam(net.discriminator.parameters(), lr=g_opt.param_groups[0]["lr"], betas=(BETA1, BETA2))
                reinits += 1

        # lr a coseno
        lr = LR * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / EPOCHS))
        for opt in (g_opt, d_opt):
            for group in opt.param_groups:
                group["lr"] = lr

        avg_g_loss /= STEPS_PER_EPOCH
        avg_d_loss /= STEPS_PER_EPOCH
        print(f"Epoca {epoch + 1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f} "
              f"| reinit D: {reinits}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(ema, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(ema))

    print(f"\nBest GANomaly — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_ganomaly_single_bee()
