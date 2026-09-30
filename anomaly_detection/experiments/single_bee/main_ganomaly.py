"""
GANomaly - Anomaly detection su singole api (single bee crops).

Addestramento: encoder-decoder-encoder avversariale sulle api sane
               (VarroaDataset, TRAIN): loss del generatore = avversariale
               (feature matching) + 50 * ricostruzione L1 + encoding; il
               discriminatore viene reinizializzato quando collassa (come nel
               paper). Adam lr 2e-4, betas (0.5, 0.999). Input in [-1, 1]
               (mean = std = 0.5) perché il decoder termina con una Tanh.
               Score immagine = ||z - z'||^2 fra i due latenti (paper); mappa =
               errore di ricostruzione per pixel smussato (gaussiano sigma 2).
Best:          ogni EVAL_INTERVAL epoche (e all'ultima) metriche sul test set;
               la configurazione migliore (img_roc_auc, a parità pxl_avg_prec)
               viene salvata in checkpoints/single_bee/ganomaly/ (best.pth +
               config.json).
Heatmap e metriche finali (AUROC, AvgPrec, F1, AUPRO) dalla best:
    python experiments/plot_heatmaps.py single_bee ganomaly

IMG_SIZE deve essere una potenza di 2: GANomaly fa padding alla potenza di 2
successiva e la mappa di ricostruzione deve coprire solo l'immagine.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.ganomaly.loss import GeneratorLoss, DiscriminatorLoss
from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker, clean_state_dict, script_config
from moviad.utilities.evaluation.report import compute_metrics, print_metrics, run_inference


# ──────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "ganomaly"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee_merged/cleaned_bees"
# i crop delle api sono 512x512: maschere GT, metriche pixel e heatmap a questa
# risoluzione per tutti i modelli (la mappa viene riportata a GT_SIZE)
GT_SIZE       = (512, 512)

IMG_SIZE      = (128, 128)
LATENT_VEC_SIZE = 100
EPOCHS        = 100
BATCH_SIZE    = 32
LR            = 2e-4
BETA1         = 0.5
BETA2         = 0.999
MAP_SIGMA     = 2            # smoothing gaussiano della mappa di ricostruzione
EVAL_INTERVAL = 10           # valuta ogni N epoche (e sempre all'ultima)
NUM_WORKERS   = 8
SEED          = 1

# Il decoder termina con Tanh: input e ricostruzione devono stare in [-1, 1]
GANOMALY_MEAN = [0.5, 0.5, 0.5]
GANOMALY_STD  = [0.5, 0.5, 0.5]

NUM_HEATMAPS  = 20           # metà sane, metà con varroa
RAW_LABEL     = "Errore di ricostruzione"
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


def build_model(device):
    return Ganomaly(
        input_size=IMG_SIZE,
        num_input_channels=3,
        n_features=64,
        latent_vec_size=LATENT_VEC_SIZE,
        extra_layers=0,
        add_final_conv_layer=True,
    ).to(device)


def model_state(model):
    return clean_state_dict(model.state_dict())


def load_model_state(model, state):
    model.load_state_dict(state)


def map_postprocess(maps, sigma=MAP_SIGMA):
    """Smoothing gaussiano dell'errore di ricostruzione (riduce il rumore per pixel)."""
    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    return TF.gaussian_blur(maps, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def train_ganomaly_single_bee():
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
    print(f"Parametri GANomaly: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

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

    # --- Optimizer e loss (training GAN custom) ---
    g_opt = Adam(model.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(model.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    g_loss_fn = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    d_loss_fn = DiscriminatorLoss()
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()))

    print(f"\n{'='*60}")
    print(f"Training GANomaly single-bee | {EPOCHS} epoche | {len(train_loader)} batch/epoca")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        model.train()
        avg_g_loss = 0.0
        avg_d_loss = 0.0

        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(device)

            # Forward pass
            padded, fake, latent_i, latent_o = model(images)
            pred_real_score, feat_real = model.discriminator(padded)

            # Generator update (feature matching sul discriminatore)
            pred_fake_score, feat_fake = model.discriminator(fake)
            g_loss = g_loss_fn(latent_i, latent_o, padded, fake, feat_real, feat_fake)

            g_opt.zero_grad()
            g_loss.backward(retain_graph=True)
            g_opt.step()
            avg_g_loss += g_loss.item()

            # Discriminator update
            pred_fake_score, _ = model.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real_score, pred_fake_score)

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            avg_d_loss += d_loss.item()

            # Se il discriminatore collassa lo si reinizializza (come nel paper)
            if d_loss.item() < 1e-5:
                model.discriminator.apply(Ganomaly.weights_init)

        avg_g_loss /= len(train_loader)
        avg_d_loss /= len(train_loader)
        print(f"Epoca {epoch + 1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(model, test_loader, device, map_postprocess)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(model))

    print(f"\nBest GANomaly — {tracker.summary()}")
    print(f"Heatmap e metriche finali: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_ganomaly_single_bee()
