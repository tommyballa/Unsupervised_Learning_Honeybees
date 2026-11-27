"""
Sweep di iperparametri per SuperSimpleNet single-bee (basato su main_supersimplenet.py).

Ogni run addestra adaptor + segdec e ogni --eval-every step calcola sul test set
la mappa (logit, alla risoluzione delle feature) e lo score della testa di
classificazione. Sulle stesse mappe vengono poi valutate varianti di
post-processing senza riaddestrare:
    sigma    blur gaussiano (in pixel a 128x128)
    score    cls (testa di classificazione) | max | topk<N> (media degli N
             pixel più alti della mappa, a 128x128)
Le metriche sono img AUROC / img AP e pixel AP a 128x128 (calcolata su GPU su
tutti i pixel del test set).

    CUDA_VISIBLE_DEVICES=0 python experiments/single_bee/sweep_supersimplenet.py --tag base
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_supersimplenet as ms  # noqa: E402
from sweep_paste import EVAL_RES, augment, blur, gpu_ap  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.utilities.configurations import Split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--layers", default="2,3", help="layer del backbone (layer<N>)")
    p.add_argument("--upscale", type=float, default=2, help="feature a upscale x la risoluzione del primo layer")
    p.add_argument("--pool", type=int, default=3, help="lato dell'average pooling sulle feature (1 = nessuno)")
    p.add_argument("--feat-norm", type=int, default=0,
                   help="standardizza le feature per canale (media/std stimate su N batch di train; 0 = no)")
    p.add_argument("--cls", type=int, default=1, help="0 = solo testa di segmentazione (niente testa cls)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--eval-bs", type=int, default=64)
    p.add_argument("--adaptor-lr", type=float, default=1e-4)
    p.add_argument("--segdec-lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--sched", default="multistep", help="multistep (0.8/0.9, gamma 0.4) | cos")
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--perlin-thr", type=float, default=0.2)
    p.add_argument("--stop-grad", type=int, default=1)
    p.add_argument("--aug", default="flip", help="'' | flip | rot (flip + rot90)")
    p.add_argument("--amp", type=int, default=1, help="fp16 su backbone e segdec")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


FEAT_NORM = None  # (media, std) per canale delle feature, con --feat-norm


def extract(model, x, upscale, amp):
    """Come UpscalingFeatureExtractor, con fattore di upscale configurabile (backbone congelato)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp)):
        feats = model.feature_extractor.feature_extractor(x)
        size = (int(feats[0].shape[-2] * upscale), int(feats[0].shape[-1] * upscale))
        fmap = torch.cat([F.interpolate(f, size=size, mode="bilinear") for f in feats], 1)
        fmap = model.feature_extractor.pooler(fmap).float()
    if FEAT_NORM is not None:
        fmap = (fmap - FEAT_NORM[0]) / FEAT_NORM[1]
    return fmap


@torch.no_grad()
def feature_stats(model, loader, device, args):
    """Media e std per canale delle feature sulle prime --feat-norm batch di train."""
    tot, tot2, n = 0.0, 0.0, 0
    for i, x in enumerate(loader):
        if i >= args.feat_norm:
            break
        f = extract(model, x.to(device).contiguous(memory_format=torch.channels_last), args.upscale, args.amp)
        tot = tot + f.double().sum((0, 2, 3)); tot2 = tot2 + (f.double() ** 2).sum((0, 2, 3))
        n += f.numel() // f.shape[1]
    mean = tot / n
    std = (tot2 / n - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.float().view(1, -1, 1, 1), std.float().view(1, -1, 1, 1)


def heads(model, feats, args):
    """Mappa e score; con --cls 0 solo la testa di segmentazione (score = None)."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
        if args.cls:
            return model.segdec(feats)
        return model.segdec.seg_head(feats), None


@torch.no_grad()
def infer(model, loader, device, args):
    model.eval()
    maps, cls, labels, masks = [], [], [], []
    for images, batch_labels, batch_masks, _ in tqdm(loader, desc="Inferenza", leave=False):
        x = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        adapted = model.adaptor(extract(model, x, args.upscale, args.amp))
        amap, score = heads(model, adapted, args)
        maps.append(amap.float().cpu())
        cls.append(score.float().reshape(-1).cpu() if score is not None else torch.zeros(len(x)))
        labels.append(np.asarray(batch_labels).reshape(-1))
        m = F.interpolate(batch_masks.float(), size=(EVAL_RES, EVAL_RES), mode="area")
        masks.append(m > 0.5)
    return torch.cat(maps), torch.cat(cls).numpy(), np.concatenate(labels), torch.cat(masks)


def variant_table(maps, cls, labels, masks, device):
    gt = masks.to(device).flatten()
    rows = []
    for sigma in (0.0, 1.0, 2.0, 4.0):
        flat_all, scores = [], {m: [] for m in ("max", "topk10", "topk50")}
        for i in range(0, len(labels), 512):
            m = F.interpolate(maps[i:i + 512].to(device), size=(EVAL_RES, EVAL_RES),
                              mode="bilinear", align_corners=False)
            flat = blur(m, sigma).flatten(1)
            scores["max"].append(flat.max(1).values.cpu())
            for k in (10, 50):
                scores[f"topk{k}"].append(flat.topk(k, dim=1).values.mean(1).cpu())
            flat_all.append(flat.half())
        pxl_ap = gpu_ap(torch.cat(flat_all).flatten().float(), gt)
        del flat_all
        named = {k: torch.cat(v).numpy() for k, v in scores.items()}
        if sigma == 0.0:
            named["cls"] = cls
        for mode, sc in named.items():
            rows.append({"sigma": sigma, "score": mode, "img_auc": roc_auc_score(labels, sc),
                         "img_ap": average_precision_score(labels, sc), "pxl_ap": pxl_ap})
    return rows


def main():
    args = parse_args()
    ms.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    ms.IMG_SIZE = (args.img_size, args.img_size)
    ms.BACKBONE = args.backbone
    ms.LAYERS = [f"layer{i}" for i in args.layers.split(",")]
    ms.PERLIN_THRESHOLD = args.perlin_thr
    ms.NOISE_STD = args.noise_std
    ms.FEATURE_UPSCALE = args.upscale
    ms.CLS_HEAD = bool(args.cls)

    log_dir = ms.REPO_ROOT / "logs" / "supersimplenet_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ms.REPO_ROOT / "checkpoints" / "single_bee" / "supersimplenet_sweep" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VarroaDataset(ms.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(ms.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_bs, shuffle=False, num_workers=12, pin_memory=True)

    model = ms.build_model(device)
    model.segdec.stop_grad = bool(args.stop_grad)
    model.feature_extractor.pooler = torch.nn.AvgPool2d(args.pool, stride=1, padding=args.pool // 2)
    model.feature_extractor.feature_extractor.model.to(memory_format=torch.channels_last)
    model.to(memory_format=torch.channels_last)

    optimizer = AdamW([
        {"params": model.adaptor.parameters(), "lr": args.adaptor_lr},
        {"params": model.segdec.parameters(), "lr": args.segdec_lr, "weight_decay": args.wd},
    ])
    scheduler = (CosineAnnealingLR(optimizer, T_max=args.steps) if args.sched == "cos"
                 else MultiStepLR(optimizer, milestones=[int(args.steps * 0.8), int(args.steps * 0.9)], gamma=0.4))
    global FEAT_NORM
    if args.feat_norm:
        FEAT_NORM = feature_stats(model, train_loader, device, args)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
    loss_fn = SSNLoss()

    config = vars(args)
    history = []
    print(f"[{args.tag}] {config}\n{len(train_loader)} step/epoca, {args.steps} step totali", flush=True)

    step, t0, tot, n = 0, time.time(), 0.0, 0
    while step < args.steps:
        for x in tqdm(train_loader, desc=f"step {step}", leave=False):
            model.train()
            x = augment(x.to(device, non_blocking=True), args.aug).contiguous(memory_format=torch.channels_last)
            b = x.shape[0]
            # adaptor e generazione delle anomalie in fp32 (il rumore è piccolo rispetto alle feature)
            adapted = model.adaptor(extract(model, x, args.upscale, args.amp))
            feats, masks, labels = model.anomaly_generator(
                adapted, torch.zeros(b, 1, *adapted.shape[-2:], device=device), torch.zeros(b, device=device))
            amap, score = heads(model, feats, args)
            amap = amap.float()
            if score is None:
                loss = loss_fn.focal_loss(amap, masks) + loss_fn.trunc_l1_loss(amap, masks)
            else:
                loss = loss_fn(pred_map=amap, pred_score=score.float(), target_mask=masks, target_label=labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tot += loss.item(); n += 1; step += 1

            if step % args.eval_every == 0 or step == args.steps:
                print(f"[{args.tag}] step {step}/{args.steps} (ep {step / len(train_loader):.2f}) "
                      f"loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
                mean_loss, tot, n = tot / n, 0.0, 0
                t0 = time.time()
                maps, cls, labels_np, gt_masks = infer(model, test_loader, device, args)
                rows = variant_table(maps, cls, labels_np, gt_masks, device)
                del maps
                state = ms.model_state(model)
                if FEAT_NORM is not None:
                    state["feat_norm"] = [t.cpu() for t in FEAT_NORM]
                torch.save(state, ckpt_dir / f"step{step}.pth")
                for key in ("img_auc", "pxl_ap"):
                    rows.sort(key=lambda r: -r[key])
                    print(f"  top varianti ({key}):")
                    for r in rows[:4]:
                        print(f"    sigma={r['sigma']} score={r['score']}: "
                              f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap={r['pxl_ap']:.4f}")
                ref = next(r for r in rows if r["score"] == "cls")
                print(f"  cls sigma=0 (come main): auc={ref['img_auc']:.4f} ap={ref['img_ap']:.4f} "
                      f"pxl_ap={ref['pxl_ap']:.4f}")
                print(f"  eval {time.time() - t0:.0f}s", flush=True)
                history.append({"step": step, "loss": mean_loss, "variants": rows})
                with open(log_dir / f"{args.tag}.json", "w") as f:
                    json.dump({"config": config, "history": history}, f)
                t0 = time.time()
            if step >= args.steps:
                break

    print(f"\n[{args.tag}] FINE", flush=True)


if __name__ == "__main__":
    main()
