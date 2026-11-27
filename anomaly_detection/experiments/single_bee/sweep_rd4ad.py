"""
Hyperparameter sweep for single-bee RD4AD (based on main_rd4ad.py).

Each run trains for a few epochs, evaluates every --eval-every steps and saves the
best checkpoint in checkpoints/single_bee/rd4ad_sweep/<tag>/. Besides the run's
"default" map, each evaluation also prints the image-level metrics
(and pixel AP at 128) of post-processing variants computed on the
same per-layer maps (layers used, blur sigma, score = max / top-k
mean), to choose the post-processing without retraining.

    CUDA_VISIBLE_DEVICES=2 python experiments/single_bee/sweep_rd4ad.py --tag base --epochs 5
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
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_rd4ad as mr  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.rd4ad.loss_functions import rd4ad_cosine_loss
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import BestTracker
from moviad.utilities.evaluation.report import print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--steps", type=int, default=0, help="total steps (0 = --epochs epochs)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--sched", default="const", help="const | cos (warmup 2%% + cosine)")
    p.add_argument("--aug", default="flip", help="none | flip | flip_rot (flip + 90° rotations)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the bn+decoder weights (0 = off)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast in training")
    p.add_argument("--eval-every", type=int, default=0, help="steps between evaluations (0 = every epoch)")
    # post-processing of the map used to select the best checkpoint
    p.add_argument("--layers", default="0,1,2")
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--score", default="max", help="max | topk<N> (mean of the N highest pixels)")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def blur(m, sigma):
    if sigma <= 0:
        return m
    k = 2 * int(4 * sigma + 0.5) + 1
    return TF.gaussian_blur(m, kernel_size=[k, k], sigma=[sigma, sigma])


def build_map(layer_maps, layers, size, sigma):
    """layer_maps: list of (B,1,h,w) per-layer 1-cos maps -> map (B,1,H,W) (sum, as in RD4AD)."""
    ups = [F.interpolate(layer_maps[l], size=size, mode="bilinear", align_corners=True) for l in layers]
    return blur(torch.stack(ups, 0).sum(0), sigma)


def image_score(m, mode):
    flat = m.flatten(1)
    if mode == "max":
        return flat.max(1).values
    k = int(mode[4:])
    return flat.topk(k, dim=1).values.mean(1)


@torch.no_grad()
def infer(model, loader, device):
    """Per-layer 1-cos maps (CPU, native resolution), labels and GT masks."""
    model.eval()
    per_layer, labels, masks = None, [], []
    for images, y, m, _ in tqdm(loader, desc="Inference", leave=False):
        x = images.to(device, non_blocking=True)
        enc = model.encoder(x)
        dec = model.decoder(model.bn(enc))
        maps = [(1 - F.cosine_similarity(d, e)).unsqueeze(1).cpu() for e, d in zip(enc, dec)]
        per_layer = [[mp] for mp in maps] if per_layer is None else [a + [mp] for a, mp in zip(per_layer, maps)]
        labels.append(np.asarray(y).reshape(-1))
        masks.append((m.numpy() > 0).astype(np.uint8))
    return [torch.cat(a) for a in per_layer], np.concatenate(labels), np.concatenate(masks)


def fast_metrics(maps128, scores, labels, masks128):
    """Image-level metrics + pixel AUROC/AP at 128x128 (subsampled to ~4M pixels)."""
    step = max(1, masks128.size // 4_000_000)
    gt, pr = masks128.ravel()[::step], maps128.ravel()[::step]
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12))
    return {"img_roc_auc": roc_auc_score(labels, scores), "img_avg_prec": average_precision_score(labels, scores),
            "img_f1": float(f1), "pxl_roc_auc": roc_auc_score(gt, pr), "pxl_avg_prec": average_precision_score(gt, pr)}


def evaluate(per_layer, layers, size, sigma, device, modes=("max",)):
    """Maps at 128x128 and scores (for each mode) for one post-processing variant."""
    s = size[0] / 256  # sigma and top-k refer to a 256x256 map
    maps, scores = [], {m: [] for m in modes}
    for i in range(0, per_layer[0].shape[0], 256):
        mp = build_map([pl[i:i + 256].to(device) for pl in per_layer], layers, size, sigma * s)
        for mode in modes:
            scaled = mode if mode == "max" else f"topk{int(int(mode[4:]) * s * s)}"
            scores[mode].append(image_score(mp, scaled).cpu().numpy())
        maps.append(F.interpolate(mp, size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy())
    return np.concatenate(maps), {m: np.concatenate(v) for m, v in scores.items()}


def variant_table(per_layer, labels, masks128, size, device):
    """Image AUROC / AP and pixel AP (at 128) for several post-processing variants."""
    n = len(per_layer)
    # reduced subset: blurring at high resolution (large σ) is very expensive
    layer_sets = [tuple(range(n)), tuple(range(1, n)), (n - 1,)]
    step = max(1, masks128.size // 2_000_000)
    gt = masks128.ravel()[::step]
    rows = []
    for layers in layer_sets:
        for sigma in (0.0, 2.0, 4.0):
            maps, scores = evaluate(per_layer, layers, size, sigma, device,
                                    ("max", "topk10", "topk100", "topk1000"))
            pxl_ap = average_precision_score(gt, maps.ravel()[::step])
            for mode, sc in scores.items():
                rows.append({"layers": layers, "sigma": sigma, "score": mode,
                             "img_auc": roc_auc_score(labels, sc),
                             "img_ap": average_precision_score(labels, sc), "pxl_ap128": pxl_ap})
    return rows


def augment(x, mode):
    if mode == "none":
        return x
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-2)
    if mode == "flip_rot":
        x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=(-2, -1))
    return x


def main():
    args = parse_args()
    mr.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    size = (args.img_size, args.img_size)
    mr.IMG_SIZE = size
    layers = tuple(int(l) for l in args.layers.split(","))

    log_dir = mr.REPO_ROOT / "logs" / "rd4ad_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mr.REPO_ROOT / "checkpoints" / "single_bee" / "rd4ad_sweep" / args.tag

    train_ds = VarroaDataset(mr.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(mr.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    model = RD4AD(backbone_name=args.backbone, input_size=size).to(device)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    params = list(model.decoder.parameters()) + list(model.bn.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.wd) \
        if args.wd > 0 else torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    total = args.steps or int(args.epochs * len(train_loader))
    if args.sched == "cos":
        warm = max(1, int(0.02 * total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    else:
        scheduler = None
    eval_every = args.eval_every or len(train_loader)
    ema = None
    if args.ema > 0:
        # EMA copy of bn+decoder, used for evaluation and checkpoints
        ema = RD4AD(backbone_name=args.backbone, input_size=size).to(device)
        ema.encoder = model.encoder
        ema.bn.load_state_dict(model.bn.state_dict())
        ema.decoder.load_state_dict(model.decoder.state_dict())
        ema_pairs = [(pe, pm) for m_e, m_m in ((ema.bn, model.bn), (ema.decoder, model.decoder))
                     for pe, pm in zip(m_e.state_dict().values(), m_m.state_dict().values())]

    config = {**vars(args), "img_size": list(size)}
    tracker = BestTracker(ckpt_dir, config)
    history = []
    print(f"[{args.tag}] {config}\n{len(train_loader)} steps/epoch, {total} total steps, eval every {eval_every}")

    step, t0, tot, n = 0, time.time(), 0.0, 0
    done = False
    while not done:
        for x in tqdm(train_loader, desc=f"step {step}", leave=False):
            model.train()
            x = augment(x.to(device, non_blocking=True), args.aug)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                enc = model.encoder(x)
                dec = model.decoder(model.bn(enc))
            loss = sum(rd4ad_cosine_loss(e.float(), d.float()) for e, d in zip(enc, dec))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            tot += loss.item(); n += 1; step += 1
            if ema is not None:
                with torch.no_grad():
                    d = min(args.ema, (1 + step) / (10 + step))
                    for pe, pm in ema_pairs:
                        if pe.dtype.is_floating_point:
                            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                        else:
                            pe.copy_(pm)

            if step % eval_every == 0 or step == total:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[{args.tag}] step {step}/{total} (ep {step / len(train_loader):.2f}) "
                      f"loss {tot / n:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)")
                tot, n = 0.0, 0
                t0 = time.time()
                eval_model = ema if ema is not None else model
                per_layer, labels, masks_full = infer(eval_model, test_loader, device)
                masks128 = F.interpolate(torch.from_numpy(masks_full).float(), size=(128, 128),
                                         mode="nearest").numpy() > 0.5
                del masks_full
                maps, scores = evaluate(per_layer, layers, size, args.sigma, device, (args.score,))
                results = fast_metrics(maps, scores[args.score], labels, masks128)
                del maps
                print_metrics(results)
                rows = variant_table(per_layer, labels, masks128, size, device)
                rows.sort(key=lambda r: -r["img_auc"])
                print("  top variants (img_auc):")
                for r in rows[:6]:
                    print(f"    layers={r['layers']} sigma={r['sigma']} score={r['score']}: "
                          f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap128={r['pxl_ap128']:.4f}")
                best_px = max(rows, key=lambda r: r["pxl_ap128"])
                print(f"    best pxl: layers={best_px['layers']} sigma={best_px['sigma']} "
                      f"pxl_ap128={best_px['pxl_ap128']:.4f}")
                print(f"  eval {time.time() - t0:.0f}s", flush=True)
                tracker.update(step, results, lambda: mr.model_state(eval_model))
                history.append({"step": step, "epoch": step / len(train_loader), "lr": lr, "metrics": results,
                                "variants": [{**r, "layers": list(r["layers"])} for r in rows]})
                with open(log_dir / f"{args.tag}.json", "w") as f:
                    json.dump({"config": config, "history": history}, f, indent=1)
                t0 = time.time()
            if step >= total:
                done = True
                break

    print(f"\n[{args.tag}] BEST — {tracker.summary()}")


if __name__ == "__main__":
    main()
