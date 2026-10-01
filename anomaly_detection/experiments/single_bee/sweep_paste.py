"""
Sweep di iperparametri per PaSTe single-bee (basato su main_paste.py).

Ogni run addestra lo student e ogni --eval-every epoche calcola sul test set
le mappe teacher/student PER LAYER (alla loro risoluzione nativa). Sulle stesse
mappe vengono poi valutate varianti di post-processing senza riaddestrare:
    layer    sottoinsieme di ad_layers usato per la mappa
    agg      sum | prod | nsum (somma di mappe normalizzate con media/std
             globali per layer, stimate su immagini di TRAIN) | psum (come
             nsum ma con media/std per posizione)
    sigma    blur gaussiano (in pixel a 128x128)
    score    max | topk<N> (media degli N pixel più alti, a 128x128)
Le metriche sono img AUROC / img AP e pixel AP a 128x128 (calcolata su GPU su
tutti i pixel del test set).

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
    p.add_argument("--bootstrap", type=int, default=0, help="-1 = nessun bootstrap (STFPM)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--opt", default="sgd", help="sgd | adam")
    p.add_argument("--lr", type=float, default=0.04)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", default="cos", help="cos | const")
    p.add_argument("--layer-weights", default="", help="pesi della loss per layer, es. 1,1,2")
    p.add_argument("--aug", default="", help="flip | rot (flip + rot90)")
    p.add_argument("--ema", type=float, default=0.0, help="decay della EMA dei pesi dello student (0 = no)")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--n-stats", type=int, default=1024, help="immagini di train per le statistiche nsum/psum")
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


@torch.no_grad()
def layer_maps(model, x, amp):
    """Mappe (B,1,h,w) di ||t/|t| - s/|s|||^2 per ogni ad_layer."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
        t_feat, bootstrap = model.teacher(x)
        s_feat, _ = model.student(x if model.student_bootstrap_layer is None else bootstrap)
    t_feat = t_feat[-len(s_feat):]
    out = []
    for t, s in zip(t_feat, s_feat):
        t = F.normalize(t.float(), dim=1)
        s = F.normalize(s.float(), dim=1)
        out.append(torch.sum((t - s) ** 2, 1, keepdim=True))
    return out


@torch.no_grad()
def infer(model, loader, device, amp, with_labels=True):
    model.eval()
    per_layer, labels, masks = None, [], []
    for batch in tqdm(loader, desc="Inferenza", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        maps = [m.half().cpu() for m in layer_maps(model, images.to(device, non_blocking=True), amp)]
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
    """Average precision (come sklearn, ignorando i pari merito) su GPU."""
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
        for agg in (("sum",) if len(idxs) == 1 else ("sum", "prod", "nsum")) + ("psum",):
            for sigma in (0.0, 1.0, 2.0, 4.0):
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
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps)
                 if args.sched == "cos" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0))

    config = vars(args)
    history = []
    print(f"[{args.tag}] {config}\n{steps} step/epoca", flush=True)

    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b, x in enumerate(tqdm(train_loader, desc=f"ep{epoch + 1}", leave=False)):
            if args.max_train_batches and b >= args.max_train_batches:
                break
            x = augment(x.to(device, non_blocking=True), args.aug).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                with torch.no_grad():
                    t_feat, bootstrap = model.teacher(x)
                s_feat, _ = model.student(x if model.student_bootstrap_layer is None else bootstrap)
            t_feat = t_feat[-len(s_feat):]
            loss = 0.0
            for w, t, s in zip(lw, t_feat, s_feat):
                t = F.normalize(t.float(), dim=1)
                s = F.normalize(s.float(), dim=1)
                loss = loss + w * (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model.student)
            tot += loss.item(); n += 1
        print(f"[{args.tag}] epoca {epoch + 1}/{args.epochs} loss {tot / n:.3f} "
              f"lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)

        if (epoch + 1) % args.eval_every and epoch != args.epochs - 1:
            continue
        t0 = time.time()
        if ema is not None:
            train_student, model.student = model.student, ema.module
        stats = compute_stats(infer(model, stats_loader, device, args.amp, with_labels=False))
        per_layer, labels, masks = infer(model, test_loader, device, args.amp)
        rows = variant_table(per_layer, labels, masks, ad_layers, stats, device)
        del per_layer
        torch.save({"student": mp.clean_state_dict(model.student.state_dict()),
                    "stats": {k: [v.cpu() if torch.is_tensor(v) else v for v in vs] for k, vs in stats.items()}},
                   ckpt_dir / f"ep{epoch + 1}.pth")
        if ema is not None:
            model.student = train_student
        for key in ("img_auc", "pxl_ap"):
            rows.sort(key=lambda r: -r[key])
            print(f"  top varianti ({key}):")
            for r in rows[:5]:
                print(f"    layers={r['layers']} agg={r['agg']} sigma={r['sigma']} score={r['score']}: "
                      f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap={r['pxl_ap']:.4f}")
        print(f"  eval {time.time() - t0:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "loss": tot / n, "variants": rows})
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": config, "history": history}, f)


if __name__ == "__main__":
    main()
