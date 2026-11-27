"""
Hyperparameter sweep for single-bee PaSTe (based on main_paste.py).

Each run trains the student and every --eval-every epochs computes on the test set
the teacher/student maps PER LAYER (at their native resolution). Post-processing
variants are then evaluated on the same maps without retraining:
    layer    subset of ad_layers used for the map
    agg      sum | prod | nsum (sum of maps normalised with global per-layer
             mean/std, estimated on TRAIN images) | psum (like
             nsum but with per-position mean/std)
    sigma    Gaussian blur (in pixels at 128x128)
    score    max | topk<N> (mean of the N highest pixels, at 128x128)
The metrics are img AUROC / img AP and pixel AP at 128x128 (computed on GPU over
all test-set pixels).

    CUDA_VISIBLE_DEVICES=1 python experiments/single_bee/sweep_paste.py --tag base --epochs 10
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
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_paste as mp  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.paste.paste import PaSTe
from moviad.utilities.configurations import Split

EVAL_RES = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--weights", default="IMAGENET1K_V1")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--ad-layers", default="1,2,3")
    p.add_argument("--bootstrap", type=int, default=0, help="-1 = no bootstrap (STFPM)")
    p.add_argument("--epochs", type=float, default=10, help="fractional values allowed")
    p.add_argument("--eval-every", type=int, default=1, help="epochs between two evaluations (if --eval-steps is 0)")
    p.add_argument("--eval-steps", type=int, default=0, help="evaluate every N steps (and always at the end)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--opt", default="sgd", help="sgd | adam")
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", default="cos", help="cos | const")
    p.add_argument("--layer-weights", default="", help="per-layer loss weights, e.g. 1,1,2")
    p.add_argument("--aug", default="", help="flip | rot (flip + rot90)")
    p.add_argument("--ema", type=float, default=0.0, help="decay of the student weight EMA (0 = no)")
    p.add_argument("--fg-loss", action="store_true", help="loss only on bee cells (black background excluded)")
    p.add_argument("--fg-map", action="store_true", help="map zeroed outside the bee")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--n-stats", type=int, default=1024, help="training images for the nsum/psum statistics")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def augment(x, mode):
    if not mode:
        return x
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-2)
    if mode == "rot":
        k = int(torch.randint(0, 4, (1,)).item())
        x = torch.rot90(x, k, dims=(-2, -1))
    return x


FG_THRESHOLD = 10 / 255   # as in main_cfa / main_patchcore
FG_CLOSING = 15
FG_DILATION = 8


def bee_foreground(images):
    """(B,1,H,W) bee mask from ImageNet-normalised images (the crop background is black)."""
    mean = torch.tensor(mp.IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(mp.IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    mask = ((images * std + mean).amax(dim=1, keepdim=True) > FG_THRESHOLD).float()
    dil = lambda m, k: F.max_pool2d(m, k, stride=1, padding=k // 2)
    mask = 1 - dil(1 - dil(mask, FG_CLOSING), FG_CLOSING)
    return dil(mask, 2 * FG_DILATION + 1)


def cell_foreground(fg, size):
    """Mask at a layer's resolution: a cell is bee if at least half of its pixels are."""
    return (F.adaptive_avg_pool2d(fg, size) >= 0.5).float()


@torch.no_grad()
def layer_maps(model, x, amp, fg_map=False):
    """Maps (B,1,h,w) of ||t/|t| - s/|s|||^2 for each ad_layer."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
        t_feat, bootstrap = model.teacher(x)
        s_feat, _ = model.student(x if model.student_bootstrap_layer is None else bootstrap)
    t_feat = t_feat[-len(s_feat):]
    out = []
    for t, s in zip(t_feat, s_feat):
        t = F.normalize(t.float(), dim=1)
        s = F.normalize(s.float(), dim=1)
        out.append(torch.sum((t - s) ** 2, 1, keepdim=True))
    if fg_map:
        fg = bee_foreground(x)
        out = [m * cell_foreground(fg, m.shape[-2:]) for m in out]
    return out


@torch.no_grad()
def infer(model, loader, device, amp, with_labels=True, fg_map=False):
    model.eval()
    per_layer, labels, masks = None, [], []
    for batch in tqdm(loader, desc="Inference", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        maps = [m.half().cpu() for m in layer_maps(model, images.to(device, non_blocking=True), amp, fg_map)]
        per_layer = [[m] for m in maps] if per_layer is None else [a + [m] for a, m in zip(per_layer, maps)]
        if with_labels:
            labels.append(np.asarray(batch[1]).reshape(-1))
            m = F.interpolate(batch[2].float(), size=(EVAL_RES, EVAL_RES), mode="area")
            masks.append(m > 0.5)
    per_layer = [torch.cat(a) for a in per_layer]
    if not with_labels:
        return per_layer
    return per_layer, np.concatenate(labels), torch.cat(masks)


def blur(m, sigma):
    if sigma <= 0:
        return m
    k = 2 * int(round(4 * sigma)) + 1
    return TF.gaussian_blur(m, kernel_size=[k, k], sigma=[sigma, sigma])


def gpu_ap(scores, gt):
    """Average precision (like sklearn, ignoring ties) on GPU."""
    order = torch.argsort(scores, descending=True)
    g = gt[order].float()
    tp = torch.cumsum(g, 0)
    prec = tp / torch.arange(1, g.numel() + 1, device=g.device, dtype=torch.float32)
    return float((prec * g).sum() / g.sum().clamp_min(1))


def build_map(per_layer, idxs, agg, stats, sigma, i0, i1, device):
    ups = []
    for j in idxs:
        m = per_layer[j][i0:i1].to(device).float()
        if agg == "nsum":
            m = (m - stats["mean"][j]) / stats["std"][j]
        elif agg == "psum":
            m = (m - stats["pmean"][j].to(device)) / stats["pstd"][j].to(device)
        ups.append(F.interpolate(m, size=(EVAL_RES, EVAL_RES), mode="bilinear", align_corners=False))
    if agg == "prod":
        out = ups[0]
        for u in ups[1:]:
            out = out * u
    else:
        out = torch.stack(ups, 0).sum(0)
    return blur(out, sigma)


def variant_table(per_layer, labels, masks, ad_layers, stats, device):
    n = len(per_layer)
    layer_sets = [c for r in range(1, n + 1) for c in itertools.combinations(range(n), r)]
    gt = masks.to(device).flatten()
    rows = []
    for idxs in layer_sets:
        for agg in (("sum",) if len(idxs) == 1 else ("sum", "prod", "nsum")):
            for sigma in (0.0, 2.0, 4.0, 8.0):
                maps, scores = [], {m: [] for m in ("max", "topk10", "topk50")}
                for i in range(0, len(labels), 512):
                    mp_ = build_map(per_layer, idxs, agg, stats, sigma, i, i + 512, device)
                    flat = mp_.flatten(1)
                    scores["max"].append(flat.max(1).values.cpu())
                    for k in (10, 50):
                        scores[f"topk{k}"].append(flat.topk(k, dim=1).values.mean(1).cpu())
                    maps.append(flat.half())
                pxl_ap = gpu_ap(torch.cat(maps).flatten().float(), gt)
                del maps
                for mode, sc in scores.items():
                    sc = torch.cat(sc).numpy()
                    rows.append({"layers": [ad_layers[j] for j in idxs], "agg": agg, "sigma": sigma,
                                 "score": mode, "img_auc": roc_auc_score(labels, sc),
                                 "img_ap": average_precision_score(labels, sc), "pxl_ap": pxl_ap})
    return rows


def compute_stats(per_layer):
    stats = {"mean": [], "std": [], "pmean": [], "pstd": []}
    for m in per_layer:
        m = m.float()
        stats["mean"].append(float(m.mean()))
        stats["std"].append(float(m.std()))
        stats["pmean"].append(m.mean(0, keepdim=True))
        stats["pstd"].append(m.std(0, keepdim=True).clamp_min(1e-6))
    return stats


def main():
    args = parse_args()
    mp.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    size = (args.img_size, args.img_size)
    mp.IMG_SIZE = size
    ad_layers = [int(x) for x in args.ad_layers.split(",")]
    lw = [float(x) for x in args.layer_weights.split(",")] if args.layer_weights else [1.0] * len(ad_layers)

    log_dir = mp.REPO_ROOT / "logs" / "paste_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mp.REPO_ROOT / "checkpoints" / "single_bee" / "paste_sweep" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VarroaDataset(mp.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(mp.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)
    stats_idx = np.random.RandomState(0).choice(len(train_ds), args.n_stats, replace=False)
    stats_loader = DataLoader(Subset(train_ds, stats_idx), batch_size=64, num_workers=12)

    model = PaSTe(backbone_model_name=args.backbone, input_size=size, output_size=size,
                  ad_layers=ad_layers, weights=args.weights,
                  student_bootstrap_layer=None if args.bootstrap < 0 else args.bootstrap).to(device)
    model.student.to(memory_format=torch.channels_last)
    model.teacher.to(memory_format=torch.channels_last)
    params = list(model.student.parameters())
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    ema = None
    if args.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            model.student, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
    steps = args.max_train_batches or len(train_loader)
    total_steps = int(round(args.epochs * steps))
    eval_steps = args.eval_steps or args.eval_every * steps
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
                 if args.sched == "cos" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0))

    config = vars(args)
    history = []
    print(f"[{args.tag}] {config}\n{steps} steps/epoch, {total_steps} total steps", flush=True)

    def batches():
        while True:
            for b, x in enumerate(train_loader):
                if args.max_train_batches and b >= args.max_train_batches:
                    break
                yield x

    data = batches()
    t0, tot, n = time.time(), 0.0, 0
    for step in tqdm(range(1, total_steps + 1), desc=args.tag, leave=False):
        model.train()
        x = augment(next(data).to(device, non_blocking=True), args.aug).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            with torch.no_grad():
                t_feat, bootstrap = model.teacher(x)
            s_feat, _ = model.student(x if model.student_bootstrap_layer is None else bootstrap)
        t_feat = t_feat[-len(s_feat):]
        loss = 0.0
        fg = bee_foreground(x) if args.fg_loss else None
        for w, t, s in zip(lw, t_feat, s_feat):
            t = F.normalize(t.float(), dim=1)
            s = F.normalize(s.float(), dim=1)
            d = (t - s) ** 2
            if fg is not None:
                d = d * cell_foreground(fg, t.shape[-2:])
            loss = loss + w * (0.5 / (t.shape[2] * t.shape[3])) * d.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_parameters(model.student)
        tot += loss.item(); n += 1

        if step % eval_steps and step != total_steps:
            continue
        epoch = round(step / steps, 3)
        mean_loss = tot / n
        print(f"[{args.tag}] step {step}/{total_steps} (epoch {epoch}) loss {mean_loss:.3f} "
              f"lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
        tot, n = 0.0, 0
        t0 = time.time()
        if ema is not None:
            train_student, model.student = model.student, ema.module
        stats = compute_stats(infer(model, stats_loader, device, args.amp, with_labels=False, fg_map=args.fg_map))
        per_layer, labels, masks = infer(model, test_loader, device, args.amp, fg_map=args.fg_map)
        rows = variant_table(per_layer, labels, masks, ad_layers, stats, device)
        del per_layer
        torch.save({"student": mp.clean_state_dict(model.student.state_dict()),
                    "stats": {k: [v.cpu() if torch.is_tensor(v) else v for v in vs] for k, vs in stats.items()}},
                   ckpt_dir / f"step{step}.pth")
        if ema is not None:
            model.student = train_student
        for key in ("img_auc", "pxl_ap"):
            rows.sort(key=lambda r: -r[key])
            print(f"  top variants ({key}):")
            for r in rows[:5]:
                print(f"    layers={r['layers']} agg={r['agg']} sigma={r['sigma']} score={r['score']}: "
                      f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap={r['pxl_ap']:.4f}")
        print(f"  eval {time.time() - t0:.0f}s", flush=True)
        history.append({"epoch": epoch, "step": step, "loss": mean_loss, "variants": rows})
        t0 = time.time()
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": config, "history": history}, f)


if __name__ == "__main__":
    main()
