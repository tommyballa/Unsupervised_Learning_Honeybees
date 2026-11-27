"""
Hyperparameter sweep for single-bee GANomaly (based on main_ganomaly.py).

For speed, train and test are fixed subsets (the same for all runs)
of train_single_bee_batch3+4 and test_single_bee_batch3+4_varroa: the images
are decoded only once, cached (uint8) and kept on GPU,
so an epoch takes a few seconds.

Every EVAL_EVERY epochs a single inference on the test subset, from which the
following are compared without redoing the forward pass:
  - the paper's image score ("latent" = ||z - z'||^2) and the distance between
    the discriminator features of input and reconstruction ("dfeat");
  - the reconstruction error maps (L1 / L2), with various blur sigmas,
    with or without the bee mask, and the max / top-k mean image scores.
Fast metrics: image-level + pixel AUROC/AP at 128x128.

    CUDA_VISIBLE_DEVICES=2 python experiments/single_bee/sweep_ganomaly.py --tag base128

Logs and results: logs/ganomaly_sweep/<tag>.{log,json}
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import gaussian_blur

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_ganomaly as mg  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.ganomaly.loss import DiscriminatorLoss
from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import clean_state_dict, save_checkpoint


SUBSET_SEED = 0             # the subsets do not depend on the training seed
SIGMAS = (0.0, 2.0, 4.0, 6.0)   # in pixels at 128x128 (scaled with the resolution)
# l1/l2: blur of the absolute error; sgn/chr: error of the smoothed signed residual
# (the texture cancels out, a compact blob remains), on RGB or on chrominance only
ERRORS = ("l1", "l2", "sgn", "chr")
FG_MODES = ("none", "ero")
SCORES = ("max", "topk10", "topk100", "mean")   # top-k in pixels at 128x128
# per-pixel maps in patch mode: (error, sigma at 128x128); for sgn/chr the sigma
# is at the varroa scale (~6 px at 128x128)
PATCH_MAPS = [("l1", 0.0), ("l1", 2.0)] + [(e, s) for e in ("sgn", "chr") for s in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
FG_THRESHOLD = 10 / 255


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--img-size", type=int, default=128, help="power of 2")
    p.add_argument("--patch", type=int, default=0,
                   help="GANomaly on PxP patches of the image at --img-size (0 = whole image)")
    p.add_argument("--stride", type=int, default=0, help="patch stride at inference (default patch/2)")
    p.add_argument("--steps", type=int, default=500, help="batches per epoch in patch mode")
    p.add_argument("--min-fg", type=float, default=0.5, help="minimum bee fraction in a patch")
    p.add_argument("--latent", type=int, default=100)
    p.add_argument("--nf", type=int, default=64)
    p.add_argument("--extra-layers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.0,
                   help="decay of the weight moving average used for evaluation (0 = current weights)")
    p.add_argument("--cosine", action="store_true", help="lr with cosine decay down to 0")
    p.add_argument("--wadv", type=float, default=1.0)
    p.add_argument("--wcon", type=float, default=50.0)
    p.add_argument("--wenc", type=float, default=1.0)
    p.add_argument("--con-loss", choices=("l1", "l2"), default="l1")
    p.add_argument("--fg-weight", type=float, default=0.0,
                   help="extra reconstruction weight on bee pixels (0 = uniform loss)")
    p.add_argument("--d-reinit-max", type=float, default=5.0,
                   help="also re-initialise the discriminator if its loss exceeds this value (saturated)")
    p.add_argument("--aug", action="store_true", help="flips and 90-degree rotations")
    p.add_argument("--train-images", type=int, default=8000)
    p.add_argument("--test-normals", type=int, default=2000)
    p.add_argument("--test-anomalies", type=int, default=1000)
    p.add_argument("--cache-dir", default=str(mg.REPO_ROOT / "checkpoints" / "single_bee" / "ganomaly_sweep" / "_cache"))
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Data: fixed cached subsets
# ──────────────────────────────────────────────────────────────────────
class _Raw(Dataset):
    def __init__(self, image_paths, mask_paths, size):
        self.image_paths, self.mask_paths, self.size = image_paths, mask_paths, size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = torch.from_numpy(np.asarray(Image.open(self.image_paths[i]).convert("RGB"))).permute(2, 0, 1)
        img = F.interpolate(img[None].float(), size=self.size, mode="bilinear", antialias=True, align_corners=False)
        img = img.round().clamp(0, 255).byte()[0]
        mask = torch.zeros(1, *self.size, dtype=torch.uint8)
        if self.mask_paths[i]:
            m = torch.from_numpy(np.asarray(Image.open(self.mask_paths[i]).convert("L")))[None, None].float()
            mask = (F.adaptive_max_pool2d(m, self.size)[0] > 127).byte()
        return img, mask


def _cached(path, image_paths, mask_paths, size):
    if path.exists():
        data = np.load(path)
        return torch.from_numpy(data["images"]), torch.from_numpy(data["masks"])
    loader = DataLoader(_Raw(image_paths, mask_paths, size), batch_size=64, num_workers=16)
    images, masks = zip(*loader)
    images, masks = torch.cat(images), torch.cat(masks)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images.numpy(), masks=masks.numpy())
    return images, masks


def load_data(args):
    """Train (N,3,S,S) uint8, test (M,3,S,S) uint8, masks (M,1,S,S), labels (M,)."""
    size = (max(args.img_size, 256),) * 2    # cache at >=256, downscaled on GPU if needed
    cache = Path(args.cache_dir)
    rng = np.random.default_rng(SUBSET_SEED)

    train_ds = VarroaDataset(mg.dataset_args(), "varroa", Split.TRAIN)
    idx = np.sort(rng.choice(len(train_ds), min(args.train_images, len(train_ds)), replace=False))
    train, _ = _cached(cache / f"train_{len(idx)}_{size[0]}.npz",
                       [train_ds.image_paths[i] for i in idx], [""] * len(idx), size)

    test_ds = VarroaDataset(mg.dataset_args(), "varroa", Split.TEST)
    labels = np.asarray(test_ds.labels)
    normal, anomal = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    idx = np.sort(np.concatenate([
        rng.choice(normal, min(args.test_normals, len(normal)), replace=False),
        rng.choice(anomal, min(args.test_anomalies, len(anomal)), replace=False),
    ]))
    test, masks = _cached(cache / f"test_{args.test_normals}_{args.test_anomalies}_{size[0]}.npz",
                          [test_ds.image_paths[i] for i in idx], [test_ds.mask_paths[i] for i in idx], size)
    return train, test, masks, labels[idx]


def to_input(batch_u8, size):
    """uint8 (B,3,H,W) -> float in [-1, 1] at the model resolution."""
    x = batch_u8.float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x * 2 - 1


def bee_mask(x):
    """(B,1,H,W) bee mask from images in [-1, 1] (the background is black)."""
    return ((x + 1) / 2).amax(dim=1, keepdim=True) > FG_THRESHOLD


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def blur(m, sigma):
    if sigma <= 0:
        return m
    return gaussian_blur(m, kernel_size=2 * int(4.0 * sigma + 0.5) + 1, sigma=sigma)


def image_score(m, mode):
    flat = m.flatten(1)
    if mode == "max":
        return flat.max(1).values
    if mode == "mean":
        return flat.mean(1)
    return flat.topk(int(mode[4:]), dim=1).values.mean(1)


def image_metrics(labels, scores):
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12))
    return {"img_auc": float(roc_auc_score(labels, scores)),
            "img_ap": float(average_precision_score(labels, scores)), "img_f1": float(f1)}


@torch.no_grad()
def evaluate(model, test, masks, labels, args, device):
    model.eval()
    scale = args.img_size / 128
    ero = 2 * int(round(3 * scale)) + 1
    latent, dfeat = [], []
    maps = {(e, s, fg): [] for e in ERRORS for s in SIGMAS for fg in FG_MODES}
    scores = {k + (m,): [] for k in maps for m in SCORES}
    for i in range(0, len(test), 256):
        x = to_input(test[i:i + 256].to(device), args.img_size)
        fake, z_i, z_o = model.generator(x)
        latent.append((z_i - z_o).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())
        _, f_real = model.discriminator(x)
        _, f_fake = model.discriminator(fake)
        dfeat.append((f_real - f_fake).pow(2).mean(dim=(1, 2, 3)).cpu().numpy())

        fg = bee_mask(x).float()
        fgs = {"none": torch.ones_like(fg),
               "ero": 1 - F.max_pool2d(1 - fg, ero, stride=1, padding=ero // 2)}
        diff = x - fake
        chroma = diff - diff.mean(1, keepdim=True)
        for e in ERRORS:
            for s in SIGMAS:
                if e == "l1":
                    blurred = blur(diff.abs().mean(1, keepdim=True), s * scale)
                elif e == "l2":
                    blurred = blur(diff.pow(2).mean(1, keepdim=True), s * scale)
                else:
                    blurred = blur(diff if e == "sgn" else chroma, s * scale).abs().mean(1, keepdim=True)
                for name, mask in fgs.items():
                    m128 = F.interpolate(blurred * mask, size=(128, 128), mode="bilinear", align_corners=False)
                    maps[(e, s, name)].append(m128.cpu().numpy())
                    for mode in SCORES:
                        scores[(e, s, name, mode)].append(image_score(m128, mode).cpu().numpy())

    gt = F.adaptive_max_pool2d(masks.float(), (128, 128)).numpy().ravel() > 0.5
    step = max(1, gt.size // 3_000_000)     # pixel metrics on a sample
    gt = gt[::step]
    rows = [{"score": "latent", **image_metrics(labels, np.concatenate(latent))},
            {"score": "dfeat", **image_metrics(labels, np.concatenate(dfeat))}]
    for (e, s, name), chunks in maps.items():
        pred = np.concatenate(chunks).ravel()[::step]
        pxl = {"pxl_auc128": float(roc_auc_score(gt, pred)), "pxl_ap128": float(average_precision_score(gt, pred))}
        for mode in SCORES:
            sc = np.concatenate(scores[(e, s, name, mode)])
            rows.append({"err": e, "sigma": s, "fg": name, "score": mode, **image_metrics(labels, sc), **pxl})
    return rows


# ──────────────────────────────────────────────────────────────────────
# Patch mode
# ──────────────────────────────────────────────────────────────────────
def sample_patches(train, args):
    """Batch of random patches (in [-1, 1]) containing enough bee."""
    n, p = 2 * args.bs, args.patch
    x = to_input(train[torch.randint(len(train), (n,), device=train.device)], args.img_size)
    oy = torch.randint(args.img_size - p + 1, (n,), device=x.device)
    ox = torch.randint(args.img_size - p + 1, (n,), device=x.device)
    ar = torch.arange(p, device=x.device)
    rows, cols = (oy[:, None] + ar)[:, :, None], (ox[:, None] + ar)[:, None, :]
    patches = x[torch.arange(n, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    keep = bee_mask(patches).float().mean(dim=(1, 2, 3)) >= args.min_fg
    return patches[keep][:args.bs].contiguous()


@torch.no_grad()
def evaluate_patches(model, test, masks, labels, args, device):
    """Per-patch scores (latent, dfeat, mean error) on a grid + reassembled per-pixel error."""
    model.eval()
    p, stride, size = args.patch, args.stride or args.patch // 2, args.img_size
    scale = size / 128
    ero = 2 * int(round(3 * scale)) + 1
    grid_names = ("latent", "dfeat", "l1patch")
    grids = {g: [] for g in grid_names}
    pix = {k: [] for k in PATCH_MAPS}
    for i in range(0, len(test), 64):
        x = to_input(test[i:i + 64].to(device), size)
        b = x.shape[0]
        patches = F.unfold(x, p, stride=stride)                      # (B, 3pp, L)
        n_side = int(round(patches.shape[-1] ** 0.5))
        patches = patches.transpose(1, 2).reshape(-1, 3, p, p)
        stats = {g: [] for g in grid_names}
        fakes = []
        for j in range(0, len(patches), 4096):
            xp = patches[j:j + 4096]
            fake, z_i, z_o = model.generator(xp)
            _, f_real = model.discriminator(xp)
            _, f_fake = model.discriminator(fake)
            err = (xp - fake).abs().mean(1, keepdim=True)
            fakes.append(fake)
            stats["latent"].append((z_i - z_o).pow(2).mean(dim=(1, 2, 3)))
            stats["dfeat"].append((f_real - f_fake).pow(2).mean(dim=(1, 2, 3)))
            stats["l1patch"].append(err.mean(dim=(1, 2, 3)))
        valid = (bee_mask(patches).float().mean(dim=(1, 2, 3)) >= args.min_fg).reshape(b, 1, n_side, n_side)
        for g in grid_names:
            grids[g].append((torch.cat(stats[g]).reshape(b, 1, n_side, n_side) * valid).cpu())
        # image reconstruction: mean of the patches covering the pixel
        fakes = torch.cat(fakes).reshape(b, -1, 3 * p * p).transpose(1, 2)
        count = F.fold(torch.ones_like(fakes), (size, size), p, stride=stride)
        diff = x - F.fold(fakes, (size, size), p, stride=stride) / count.clamp(min=1)
        chroma = diff - diff.mean(1, keepdim=True)
        fg = bee_mask(x).float()
        fg = 1 - F.max_pool2d(1 - fg, ero, stride=1, padding=ero // 2)
        for e, sigma in pix:
            if e == "l1":
                m = blur(diff.abs().mean(1, keepdim=True), sigma * scale)
            else:
                m = blur(diff if e == "sgn" else chroma, sigma * scale).abs().mean(1, keepdim=True)
            pix[(e, sigma)].append(F.interpolate(m * fg, size=(128, 128), mode="bilinear", align_corners=False).cpu())

    gt = F.adaptive_max_pool2d(masks.float(), (128, 128)).numpy().ravel() > 0.5
    step = max(1, gt.size // 3_000_000)
    gt = gt[::step]

    def pixel_metrics(m128):
        pred = m128.numpy().ravel()[::step]
        return {"pxl_auc128": float(roc_auc_score(gt, pred)), "pxl_ap128": float(average_precision_score(gt, pred))}

    rows = []
    for g in grid_names:
        grid = torch.cat(grids[g])
        pxl = pixel_metrics(F.interpolate(grid, size=(128, 128), mode="bilinear", align_corners=False))
        flat = grid.flatten(1)
        variants = {"max": flat.max(1).values, "top3": flat.topk(3, dim=1).values.mean(1),
                    "mean": flat.sum(1) / (flat > 0).sum(1).clamp(min=1)}
        for mode, sc in variants.items():
            rows.append({"score": f"{g}-{mode}", **image_metrics(labels, sc.numpy()), **pxl})
    for (e, sigma), chunks in pix.items():
        m128 = torch.cat(chunks)
        pxl = pixel_metrics(m128)
        for mode in SCORES:
            rows.append({"err": e, "sigma": sigma, "fg": "ero", "score": mode,
                         **image_metrics(labels, image_score(m128, mode).numpy()), **pxl})
    return rows


def describe(r):
    name = r["score"] if "err" not in r else f"{r['err']} sigma={r['sigma']} fg={r['fg']} {r['score']}"
    pxl = f" pxl_auc128={r['pxl_auc128']:.4f} pxl_ap128={r['pxl_ap128']:.4f}" if "pxl_ap128" in r else ""
    return f"{name}: auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} f1={r['img_f1']:.4f}{pxl}"


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    mg.set_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    config = vars(args)
    print(f"[{args.tag}] {config}", flush=True)

    log_dir = mg.REPO_ROOT / "logs" / "ganomaly_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mg.REPO_ROOT / "checkpoints" / "single_bee" / "ganomaly_sweep" / args.tag

    t0 = time.time()
    train, test, masks, labels = load_data(args)
    train = train.to(device)
    print(f"[{args.tag}] train {tuple(train.shape)} | test {tuple(test.shape)} "
          f"({int((labels == 0).sum())} healthy, {int(labels.sum())} varroa) | {time.time() - t0:.0f}s", flush=True)

    model = Ganomaly(
        input_size=(args.patch or args.img_size,) * 2, num_input_channels=3, n_features=args.nf,
        latent_vec_size=args.latent, extra_layers=args.extra_layers, add_final_conv_layer=True,
    ).to(device)
    g_opt = Adam(model.generator.parameters(), lr=args.lr, betas=(mg.BETA1, mg.BETA2))
    d_opt = Adam(model.discriminator.parameters(), lr=args.lr, betas=(mg.BETA1, mg.BETA2))
    d_loss_fn = DiscriminatorLoss()

    ema = copy.deepcopy(model).requires_grad_(False) if args.ema > 0 else None
    history, best = [], None
    steps = args.steps if args.patch else len(train) // args.bs
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train), device=device)
        sums = np.zeros(5)
        reinits = 0
        for step in range(steps):
            if args.patch:
                x = sample_patches(train, args)
            else:
                x = to_input(train[perm[step * args.bs:(step + 1) * args.bs]], args.img_size)
            if args.aug:
                if torch.rand(1).item() < 0.5:
                    x = x.flip(-1)
                x = torch.rot90(x, int(torch.randint(4, (1,)).item()), dims=(-2, -1))

            fake, z_i, z_o = model.generator(x)
            pred_real, feat_real = model.discriminator(x)
            _, feat_fake = model.discriminator(fake)

            diff = (x - fake).abs() if args.con_loss == "l1" else (x - fake).pow(2)
            if args.fg_weight > 0:
                diff = diff * (1 + args.fg_weight * bee_mask(x).float())
            l_con = diff.mean()
            l_adv = F.mse_loss(feat_fake, feat_real.detach())
            l_enc = F.smooth_l1_loss(z_o, z_i)
            g_loss = args.wadv * l_adv + args.wcon * l_con + args.wenc * l_enc
            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            if ema is not None:
                # in the first iterations the average follows the current weights (decay warmup)
                seen = (epoch - 1) * steps + step + 1
                decay = min(args.ema, (1 + seen) / (10 + seen))
                for p_ema, p_cur in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p_cur.detach(), 1 - decay)
                for b_ema, b_cur in zip(ema.buffers(), model.buffers()):
                    b_ema.copy_(b_cur)

            pred_fake, _ = model.discriminator(fake.detach())
            d_loss = d_loss_fn(pred_real, pred_fake)
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            # discriminator collapsed (loss ~0) or saturated (BCE at maximum, zero
            # gradient): it is re-initialised together with its optimizer (Adam's moments from the
            # old weights would push it straight back into saturation)
            if d_loss.item() < 1e-5 or d_loss.item() > args.d_reinit_max:
                model.discriminator.apply(Ganomaly.weights_init)
                d_opt = Adam(model.discriminator.parameters(), lr=args.lr, betas=(mg.BETA1, mg.BETA2))
                reinits += 1
            sums += [g_loss.item(), d_loss.item(), l_adv.item(), l_con.item(), l_enc.item()]

        if args.cosine:
            for opt in (g_opt, d_opt):
                for group in opt.param_groups:
                    group["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        g, d, adv, con, enc = sums / steps
        print(f"[{args.tag}] ep {epoch}/{args.epochs} G={g:.4f} D={d:.4f} adv={adv:.4f} con={con:.4f} "
              f"enc={enc:.4f} reinit={reinits} ({time.time() - t0:.0f}s)", flush=True)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rows = (evaluate_patches if args.patch else evaluate)(ema or model, test, masks, labels, args, device)
            top = sorted(rows, key=lambda r: r["img_auc"], reverse=True)
            for r in (rows[:9:3] if args.patch else rows[:2]) + top[:5]:
                print(f"    {describe(r)}", flush=True)
            top_pxl = max((r for r in rows if "pxl_ap128" in r), key=lambda r: r["pxl_ap128"])
            print(f"    best pixel — {describe(top_pxl)}", flush=True)
            history.append({"epoch": epoch, "losses": {"g": g, "d": d, "adv": adv, "con": con, "enc": enc}, "rows": rows})
            if best is None or top[0]["img_auc"] > best["img_auc"]:
                best = {"epoch": epoch, **top[0]}
                save_checkpoint(ckpt_dir, clean_state_dict((ema or model).state_dict()), config, best_epoch=epoch, metrics=top[0])
            with open(log_dir / f"{args.tag}.json", "w") as f:
                json.dump({"config": config, "best": best, "history": history}, f, indent=1)

    print(f"\n[{args.tag}] BEST img_auc — epoch {best['epoch']}: {describe(best)}")


if __name__ == "__main__":
    main()
