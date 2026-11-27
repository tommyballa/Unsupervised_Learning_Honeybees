"""
Sweep di iperparametri per FastFlow single-bee (basato su main_fastflow.py).

Ogni run addestra per pochi step (1 epoca = ~1450 step con 23k immagini),
valuta ai --eval-steps e salva la best in checkpoints/single_bee/fastflow_sweep/<tag>/.
Oltre alla mappa "di default" della run, per ogni valutazione vengono
stampate le metriche image-level (e pixel AP a 256) di varianti di
post-processing calcolate sulle stesse z (scale usate, sigma del blur,
score = max / media top-k), per scegliere il post-processing senza riaddestrare.

    CUDA_VISIBLE_DEVICES=0 python experiments/single_bee/sweep_fastflow.py --tag base --steps 1500 --masked-loss 4 --fg-dilate 0 --sigma 12
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_fastflow as mf  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.fastflow.fastflow import CompleteFastFlowModel, FastflowModel, gaussian_blur, pixel_log_jacobian as pixel_log_jac
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker
from moviad.utilities.evaluation.report import compute_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=1500, help="step di training totali")
    p.add_argument("--eval-steps", default="250,500,1000,1500", help="step a cui valutare")
    p.add_argument("--full-variants", action="store_true", help="tabella completa di varianti (lenta)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--hidden-ratio", type=float, default=1.0)
    p.add_argument("--conv3x3-only", action="store_true")
    p.add_argument("--norm", default="affine",
                   help="affine (paper) | plain (LayerNorm senza affine) | channel (LN sui soli canali) | none")
    p.add_argument("--clip-grad", type=float, default=0.0)
    p.add_argument("--hflip", action="store_true", help="augment: flip orizzontale/verticale in training")
    # post-processing della mappa usata per la selezione della best
    p.add_argument("--scales", default="0,1,2", help="indici delle scale usate per la mappa")
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--map", default="z2", help="z2 (FastFlow: mean z^2) | nll (0.5 z^2 - log-det, per canale)")
    p.add_argument("--score", default="max", help="max | topk<N> (media dei N pixel più alti)")
    p.add_argument("--fg-dilate", type=int, default=-99,
                   help="mappa limitata all'ape (pixel non neri) dilatata di N px (negativo = erosione); -99 = nessuna maschera")
    p.add_argument("--masked-loss", type=int, default=-99,
                   help="NLL per pixel mediata solo sui pixel dell'ape dilatati di N px (a 256); -99 = loss standard")
    p.add_argument("--eval-ckpt", default="", help="solo valutazione di un best.pth (nessun training)")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


class ChannelLayerNorm(torch.nn.Module):
    """LayerNorm sui canali di ogni posizione (invariante alla posizione dell'ape)."""

    def __init__(self, channels):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def build_model(args, device):
    size = (args.img_size, args.img_size)
    model = CompleteFastFlowModel(args.backbone, input_size=size, normalize=args.norm != "none", device=device)
    if args.norm == "channel":
        model.norms = torch.nn.ModuleList(ChannelLayerNorm(c) for c in model.channels)
    elif args.norm == "plain":
        model.norms = torch.nn.ModuleList(
            torch.nn.LayerNorm(list(n.normalized_shape), elementwise_affine=False) for n in model.norms)
    model.fast_flow_module = FastflowModel(
        input_size=size, flow_steps=args.flow_steps, conv3x3_only=args.conv3x3_only,
        hidden_ratio=args.hidden_ratio, channels=model.channels, scales=model.scales,
    )
    model.to(device)
    return model


def features(model, x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = model.feature_extractor(x)
    return [model.norms[i](f.float()) for i, f in enumerate(feats)]


def blur(m, sigma):
    if sigma <= 0:
        return m
    return gaussian_blur(m, sigma)


def masked_fastflow_loss(model, z, fg, dilate):
    """NLL media per pixel (per dimensione) sui soli pixel dell'ape."""
    mask = fg.float()
    if dilate > 0:
        mask = F.max_pool2d(mask, 2 * dilate + 1, stride=1, padding=dilate)
    loss = 0
    for zi, blk in zip(z, model.fast_flow_module.fast_flow_blocks):
        m = F.adaptive_max_pool2d(mask, zi.shape[-2:])[:, 0]
        nll = 0.5 * (zi ** 2).sum(1) - pixel_log_jac(blk)
        loss = loss + (nll * m).sum() / (m.sum() * zi.shape[1]).clamp_min(1)
    return loss


FG_THR = 0.1  # intensità massima RGB (0-1) sopra cui un pixel è ape e non sfondo nero


def foreground(images):
    """Maschera (B,1,H,W) dei pixel dell'ape: lo sfondo dei crop è nero."""
    mean = torch.tensor(mf.IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(mf.IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return ((images * std + mean).amax(1, keepdim=True) > FG_THR).to(torch.uint8)


def build_map(scale_maps, scales, size, sigma, fg=None, dilate=-1):
    """scale_maps: lista (B,1,h,w) di mappe per scala -> mappa (B,1,H,W).
    Con dilate > -99 la mappa viene limitata all'ape (dilatata/erosa) prima del blur."""
    ups = [F.interpolate(scale_maps[s], size=size, mode="bilinear", align_corners=False) for s in scales]
    m = torch.stack(ups, 0).mean(0)
    if fg is not None and dilate > -99:
        mask = fg.float()
        if dilate > 0:
            mask = F.max_pool2d(mask, 2 * dilate + 1, stride=1, padding=dilate)
        elif dilate < 0:
            mask = 1 - F.max_pool2d(1 - mask, -2 * dilate + 1, stride=1, padding=-dilate)
        # fuori dall'ape la mappa vale il minimo sull'ape (la NLL può essere negativa)
        low = m.masked_fill(mask == 0, float("inf")).amin(dim=(2, 3), keepdim=True)
        low = torch.where(torch.isinf(low), torch.zeros_like(low), low)
        m = torch.where(mask > 0, m, low)
    return blur(m, sigma)


def image_score(m, mode):
    flat = m.flatten(1)
    if mode == "max":
        return flat.max(1).values
    k = int(mode[4:])
    return flat.topk(k, dim=1).values.mean(1)


@torch.no_grad()
def infer(model, loader, device, args):
    """Restituisce le mappe per scala (CPU, risoluzione nativa), label e maschere a 256."""
    model.eval()
    per_scale, labels, fgs, masks_full = None, [], [], []
    for images, y, m, _ in tqdm(loader, desc="Inferenza", leave=False):
        images = images.to(device, non_blocking=True)
        fgs.append(foreground(images).cpu())
        z, _ = model.fast_flow_module(features(model, images))
        maps = [torch.mean(zi.float() ** 2, dim=1, keepdim=True).cpu() for zi in z]
        maps += [(0.5 * torch.mean(zi.float() ** 2, dim=1) - pixel_log_jac(blk) / zi.shape[1])[:, None].cpu()
                 for zi, blk in zip(z, model.fast_flow_module.fast_flow_blocks)]
        per_scale = [[mp] for mp in maps] if per_scale is None else [a + [mp] for a, mp in zip(per_scale, maps)]
        labels.append(np.asarray(y).reshape(-1))
        masks_full.append((m.numpy() > 0).astype(np.uint8))
    per_scale = [torch.cat(a) for a in per_scale]
    n = len(per_scale) // 2
    per_scale = {"z2": per_scale[:n], "nll": per_scale[n:]}
    labels = np.concatenate(labels)
    masks_full = np.concatenate(masks_full)
    return per_scale, torch.cat(fgs), labels, masks_full


def variant_table(per_scale, fg, labels, masks_full, size, device, full=False):
    """AUROC / AP image e AP pixel (a 256) per diverse varianti di post-processing."""
    n_scales = len(per_scale["z2"])
    masks_small = F.interpolate(torch.from_numpy(masks_full).float(), size=size, mode="nearest").numpy() > 0.5
    if full:
        grid = itertools.product([tuple(range(n_scales))] + list(itertools.combinations(range(n_scales), 2)),
                                 (4.0, 8.0, 12.0), (-99, -4, 0, 4))
    else:
        f = size[0] / 256  # sigma ed erosione in pixel del modello, scalati con la risoluzione
        # attorno all'ottimo sul dataset finale SAM3 (z2, tutte le scale, sigma 16, erosione 24 a 512 px)
        grid = itertools.product([tuple(range(n_scales))], (6.0 * f, 8.0 * f, 10.0 * f), (int(-8 * f), int(-12 * f)))
    rows = []
    kinds = ("z2", "nll") if full else ("z2",)
    for (scales, sigma, dilate), kind in itertools.product(list(grid), kinds):
        if True:
            maps, scores = [], {m: [] for m in ("max", "topk100")}
            for i in range(0, len(labels), 256):
                sm = [ps[i:i + 256].to(device) for ps in per_scale[kind]]
                mp = build_map(sm, scales, size, sigma, fg[i:i + 256].to(device), dilate)
                for mode in scores:
                    scores[mode].append(image_score(mp, mode).cpu().numpy())
                st = size[0] // 128
                maps.append(mp.cpu().numpy()[:, :, ::st, ::st])  # AP pixel a 128 (sottocampionata)
            maps = np.concatenate(maps)
            ms = masks_small[:, :, ::st, ::st]
            step = max(1, ms.size // 4_000_000)
            pxl_ap = average_precision_score(ms.ravel()[::step], maps.ravel()[::step])
            for mode, sc in scores.items():
                sc = np.concatenate(sc)
                rows.append({"map": kind, "scales": scales, "sigma": sigma, "fg_dilate": dilate, "score": mode,
                             "img_auc": roc_auc_score(labels, sc),
                             "img_ap": average_precision_score(labels, sc), "pxl_ap128": pxl_ap})
    return rows


def evaluate(model, test_loader, device, args, scales, size):
    t0 = time.time()
    per_scale, fg, labels, masks_full = infer(model, test_loader, device, args)
    # mappa della run alla risoluzione GT (come run_inference) per le metriche ufficiali
    maps, scores = [], []
    for i in range(0, len(labels), 128):
        mp = build_map([ps[i:i + 128].to(device) for ps in per_scale[args.map]], scales, size, args.sigma,
                       fg[i:i + 128].to(device), args.fg_dilate)
        scores.append(image_score(mp, args.score).cpu().numpy())
        maps.append(F.interpolate(mp, size=masks_full.shape[-2:], mode="bilinear",
                                  align_corners=False).cpu().numpy())
    results = compute_metrics(np.concatenate(maps), np.concatenate(scores), labels, masks_full,
                              with_pro=False)
    del maps
    print_metrics(results)
    rows = variant_table(per_scale, fg, labels, masks_full, size, device, args.full_variants)
    rows = [{**r, "scales": list(r["scales"])} for r in rows]

    def fmt(r):
        return (f"map={r['map']} scales={r['scales']} sigma={r['sigma']} fg={r['fg_dilate']} score={r['score']}: "
                f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap128={r['pxl_ap128']:.4f}")
    print("  top varianti (img_auc):")
    for r in sorted(rows, key=lambda r: -r["img_auc"])[:6]:
        print("    " + fmt(r))
    print("  top varianti (pxl_ap128):")
    for r in sorted(rows, key=lambda r: -r["pxl_ap128"])[:3]:
        print("    " + fmt(r))
    print(f"  eval {time.time() - t0:.0f}s")
    return results, rows


def main():
    args = parse_args()
    mf.set_seed(args.seed)
    device = torch.device("cuda:0")
    size = (args.img_size, args.img_size)
    mf.IMG_SIZE = size
    scales = tuple(int(s) for s in args.scales.split(","))

    log_dir = mf.REPO_ROOT / "logs" / "fastflow_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mf.REPO_ROOT / "checkpoints" / "single_bee" / "fastflow_sweep" / args.tag

    train_ds = VarroaDataset(mf.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(mf.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    model = build_model(args, device)
    if args.eval_ckpt:
        if args.eval_ckpt != "init":  # "init" = flow non addestrato (riferimento)
            model.load_state_dict(torch.load(args.eval_ckpt, map_location=device))
        results, rows = evaluate(model, test_loader, device, args, scales, size)
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": vars(args), "metrics": results, "variants": rows}, f, indent=1)
        return
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10, final_div_factor=100)
    from moviad.models.fastflow.loss_functions import fastflow_loss
    # verifica: la somma del log-det per pixel coincide con quello del flow
    with torch.no_grad():
        xb = next(iter(test_loader))[0][:2].to(device)
        z, jac = model.fast_flow_module(features(model, xb))
        for j, blk in zip(jac, model.fast_flow_module.fast_flow_blocks):
            pj = pixel_log_jac(blk).flatten(1).sum(1)
            assert torch.allclose(pj, j, rtol=1e-3), (pj, j)

    config = {**vars(args), "img_size": list(size)}
    tracker = BestTracker(ckpt_dir, config)
    history = []
    eval_steps = {int(v) for v in args.eval_steps.split(",")} | {args.steps}
    print(f"[{args.tag}] {config}\n{len(train_loader)} step/epoca")

    def batches():
        while True:
            yield from train_loader

    model.train()
    t0, tot, n = time.time(), 0.0, 0
    for step, x in enumerate(tqdm(batches(), total=args.steps, leave=False), start=1):
        x = x.to(device, non_blocking=True)
        if args.hflip:
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            if torch.rand(1).item() < 0.5:
                x = x.flip(-2)
        z, jac = model.fast_flow_module(features(model, x))
        if args.masked_loss > -99:
            loss = masked_fastflow_loss(model, z, foreground(x), args.masked_loss)
        else:
            loss = fastflow_loss(z, jac)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        optimizer.step()
        scheduler.step()
        tot += loss.item(); n += 1

        if step in eval_steps:
            print(f"[{args.tag}] step {step}/{args.steps} loss {tot / n:.3f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
            results, rows = evaluate(model, test_loader, device, args, scales, size)
            tracker.update(step, results, lambda: mf.model_state(model))
            history.append({"step": step, "loss": tot / n, "metrics": results, "variants": rows})
            with open(log_dir / f"{args.tag}.json", "w") as f:
                json.dump({"config": config, "history": history}, f, indent=1)
            model.train()
            t0, tot, n = time.time(), 0.0, 0
        if step >= args.steps:
            break

    print(f"\n[{args.tag}] BEST — {tracker.summary()}")


if __name__ == "__main__":
    main()
