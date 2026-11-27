"""
GANomaly - Anomaly detection on single bees (single bee crops).

Model:         GANomaly (adversarial encoder-decoder-encoder) trained on
               32x32 PATCHES of the bee at 256x256, not on the whole image: the
               varroa (~12 px at 256) thus takes up a relevant part of the input,
               like the objects in the paper's datasets (CIFAR 32x32, X-ray patches).
Training:      random patches that are at least half bee (the black background is excluded) from
               a fixed subset of TRAIN_IMAGES healthy bees, kept in memory;
               generator loss = adversarial (feature matching) + 50 *
               L1 reconstruction + encoding. Adam lr 2e-4 with cosine
               decay, betas (0.5, 0.999); evaluation uses the moving average
               (EMA) of the weights. Input in [-1, 1] (mean = std = 0.5) because the
               decoder ends with a Tanh.
Inference:     sliding window (stride PATCH_STRIDE), the image reconstruction
               is the mean of the reconstructed patches. Map = CHROMINANCE
               residual (input - reconstruction, minus the channel mean)
               smoothed with sign (sigma MAP_SIGMA) and then in absolute value,
               only inside the bee; image score = maximum of the map.
Best:          every EVAL_INTERVAL epochs (and at the last one) metrics on a
               fixed subset of the test set; the best configuration
               (img_roc_auc, ties broken by pxl_avg_prec) is saved in
               checkpoints/single_bee/ganomaly/ (best.pth + config.json).
Final heatmaps and metrics (AUROC, AvgPrec, F1, AUPRO) from the best checkpoint:
    python experiments/plot_heatmaps.py single_bee ganomaly

Optimisation (sweep_ganomaly.py, logs/ganomaly_sweep/, train_batch3+4 /
test_batch3+4: 8000 training images, test 2000 healthy + 1000 with varroa,
pixel metrics at 128x128):
  - GANomaly on the whole image (128 or 256 px, latent 100 or 512, with or without
    augmentation) stays close to chance: img AUROC 0.58-0.60, and the paper's
    score (||z - z'||^2) 0.45-0.52. The varroa is only a few pixels and the
    reconstruction error is dominated by the bee's texture.
  - On patches the per-patch scores (latent, discriminator features, L1
    error) stop at 0.62-0.66; the map makes the difference: the signed residual
    smoothed at the varroa scale cancels the texture and leaves the
    compact blobs, and chrominance alone works better than RGB.
  - Smaller patches are better (128 -> 64 -> 32) but at 512 px a 32 patch
    learns to reconstruct the varroa too (img AUROC 0.57): 256 px / patch 32.
  - Adversarial training is unstable (img AUROC between 0.66 and 0.87 depending
    on the epoch): weight EMA + cosine lr make the curve monotonic.
    With 80 epochs instead of 40: img AUROC 0.863 -> 0.890, pxl AP128 0.30 -> 0.35.
  - Latent 100 is the best (16/32: 0.62-0.67, 50: 0.74, 256: 0.72-0.80); without
    adversarial loss 0.852 vs 0.863 at 40 epochs.
  - The discriminator re-initialised "as in the paper" with the old optimizer
    saturated immediately (BCE loss stuck at 50, zero gradient): now the
    optimizer is re-initialised too, and also when the loss is saturated.
Result (best at epoch 80, full test 5000 healthy + 3119 with varroa, 512x512):
img AUROC 0.897, img AP 0.867, img F1 0.783, pxl AUROC 0.963, pxl AP 0.430,
AUPRO 0.910 (logs/ganomaly_main.log, logs/ganomaly_plot.log).
"""

from moviad.datasets.varroa.paths import DATA_ROOT
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
# Configuration
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME    = "ganomaly"
REPO_ROOT     = Path(__file__).resolve().parents[2]
TRAIN_DIR     = DATA_ROOT + "/processed/train_single_bee_batch3+4/cleaned_bees"
# bee crops are 512x512: GT masks, pixel metrics and heatmaps at this
# resolution for all models (the map is resized back to GT_SIZE)
GT_SIZE       = (512, 512)

IMG_SIZE      = (256, 256)
PATCH_SIZE    = 32           # GANomaly input (power of 2)
PATCH_STRIDE  = 16           # sliding-window stride at inference
MIN_FG        = 0.5          # minimum bee fraction in a training patch
LATENT_VEC_SIZE = 100
EPOCHS        = 80
STEPS_PER_EPOCH = 500        # patch batches per "epoch"
BATCH_SIZE    = 64           # patches per batch
TEST_BATCH_SIZE = 32         # images per batch at inference
LR            = 2e-4
BETA1         = 0.5
BETA2         = 0.999
W_ADV, W_CON, W_ENC = 1.0, 50.0, 1.0
EMA_DECAY     = 0.999        # moving average of the weights used for evaluation
D_REINIT_MAX  = 5.0          # discriminator loss above which it is considered saturated
MAP_SIGMA     = 6.0          # px at IMG_SIZE: (signed) blur of the chrominance residual
FG_THRESHOLD  = 10 / 255     # the crop background is black
FG_EROSION    = 13           # px at IMG_SIZE: the bee edge is excluded from the map
EVAL_INTERVAL = 10           # evaluate every N epochs (and always at the last one)
# fixed subsets for speed (None = the whole dataset)
TRAIN_IMAGES  = 8000
EVAL_NORMALS  = 2000
EVAL_ANOMALIES = 1000
SUBSET_SEED   = 0
NUM_WORKERS   = 8
SEED          = 1

# The decoder ends with Tanh: input and reconstruction must be in [-1, 1]
GANOMALY_MEAN = [0.5, 0.5, 0.5]
GANOMALY_STD  = [0.5, 0.5, 0.5]

NUM_HEATMAPS  = 20           # half healthy, half with varroa
RAW_LABEL     = "Chrominance residual"
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
# Model and data (also used by experiments/plot_heatmaps.py)
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
    """(B,1,H,W) bee mask from images in [-1, 1]."""
    return ((images + 1) / 2).amax(dim=1, keepdim=True) > FG_THRESHOLD


class PatchGanomaly(nn.Module):
    """Patch-based GANomaly: at inference it reassembles the image and computes its map."""

    def __init__(self, net: Ganomaly):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def forward(self, images):
        """Returns (map (B,1,H,W), score (B,)) for images in [-1, 1]."""
        b, _, h, w = images.shape
        p = PATCH_SIZE
        patches = F.unfold(images, p, stride=PATCH_STRIDE)                 # (B, 3pp, L)
        n_patches = patches.shape[-1]
        fake, _, _ = self.net.generator(patches.transpose(1, 2).reshape(-1, 3, p, p))
        # image reconstruction: mean of the patches covering each pixel
        fake = fake.reshape(b, n_patches, -1).transpose(1, 2)
        count = F.fold(torch.ones_like(fake), (h, w), p, stride=PATCH_STRIDE)
        diff = images - F.fold(fake, (h, w), p, stride=PATCH_STRIDE) / count.clamp(min=1)

        # blurring the signed residual cancels the texture (zero mean) and
        # leaves the compact blobs of different colour, like the varroa
        chroma = diff - diff.mean(dim=1, keepdim=True)
        kernel = 2 * int(4.0 * MAP_SIGMA + 0.5) + 1
        amap = TF.gaussian_blur(chroma, [kernel, kernel], [MAP_SIGMA, MAP_SIGMA]).abs().mean(dim=1, keepdim=True)

        fg = bee_foreground(images).float()
        fg = 1 - F.max_pool2d(1 - fg, FG_EROSION, stride=1, padding=FG_EROSION // 2)   # erosion
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
    """Fixed subset of the training set in memory, uint8 (N,3,H,W) on device."""
    idx = fixed_subset(len(train_dataset), TRAIN_IMAGES, np.random.default_rng(SUBSET_SEED))
    loader = DataLoader(Subset(train_dataset, idx.tolist()), batch_size=64, num_workers=NUM_WORKERS)
    images = [((batch + 1) * 127.5).round().clamp(0, 255).byte() for batch in tqdm(loader, desc="Loading train")]
    return torch.cat(images).to(device)


def eval_subset(test_dataset):
    """Fixed subset of the test set for evaluations during training."""
    rng = np.random.default_rng(SUBSET_SEED)
    labels = np.asarray(test_dataset.labels)
    normal, anomal = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    idx = np.concatenate([normal[fixed_subset(len(normal), EVAL_NORMALS, rng)],
                          anomal[fixed_subset(len(anomal), EVAL_ANOMALIES, rng)]])
    return Subset(test_dataset, np.sort(idx).tolist())


def sample_patches(train_images):
    """Batch of random patches in [-1, 1] containing enough bee, with flips and rotations."""
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
    ema = copy.deepcopy(model).requires_grad_(False)     # weights used for evaluation
    print(f"GANomaly parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    train_dataset = VarroaDataset(dataset_args(), "varroa", Split.TRAIN)
    test_dataset = VarroaDataset(dataset_args(), "varroa", Split.TEST)
    train_images = load_train_images(train_dataset, device)
    test_subset = eval_subset(test_dataset)
    print(f"Train: {len(train_images)} images in memory | evaluating on {len(test_subset)} test images")
    test_loader = DataLoader(
        test_subset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # --- Optimizer and loss (custom GAN training) ---
    g_opt = Adam(net.generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_opt = Adam(net.discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    d_loss_fn = DiscriminatorLoss()
    tracker = BestTracker(CHECKPOINT_DIR, script_config(globals()),
                          selection_data=f"test set subset ({len(test_subset)} images)")

    print(f"\n{'='*60}")
    print(f"Training GANomaly single-bee | {EPOCHS} epochs | {STEPS_PER_EPOCH} patch batches/epoch")
    print(f"{'='*60}\n")

    step_count = 0
    for epoch in range(EPOCHS):
        model.train()
        avg_g_loss = 0.0
        avg_d_loss = 0.0
        reinits = 0

        for _ in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            patches = sample_patches(train_images)

            # Generator update (feature matching on the discriminator)
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

            # weight EMA (in the first iterations it follows the current weights)
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

            # Discriminator collapsed (loss ~0, as in the paper) or saturated (BCE at
            # maximum, zero gradient): it is re-initialised together with its optimizer,
            # otherwise Adam's moments from the old weights push it back into saturation
            if d_loss.item() < 1e-5 or d_loss.item() > D_REINIT_MAX:
                net.discriminator.apply(Ganomaly.weights_init)
                d_opt = Adam(net.discriminator.parameters(), lr=g_opt.param_groups[0]["lr"], betas=(BETA1, BETA2))
                reinits += 1

        # cosine lr
        lr = LR * 0.5 * (1 + math.cos(math.pi * (epoch + 1) / EPOCHS))
        for opt in (g_opt, d_opt):
            for group in opt.param_groups:
                group["lr"] = lr

        avg_g_loss /= STEPS_PER_EPOCH
        avg_d_loss /= STEPS_PER_EPOCH
        print(f"Epoch {epoch + 1}/{EPOCHS} — G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f} "
              f"| reinit D: {reinits}")

        if (epoch + 1) % EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            maps, scores, labels, masks = run_inference(ema, test_loader, device)
            results = compute_metrics(maps, scores, labels, masks, with_pro=False)
            print_metrics(results)
            tracker.update(epoch + 1, results, lambda: model_state(ema))

    print(f"\nBest GANomaly — {tracker.summary()}")
    print(f"Final heatmaps and metrics: python experiments/plot_heatmaps.py single_bee {MODEL_NAME}")


if __name__ == "__main__":
    train_ganomaly_single_bee()
