"""
Sweep di iperparametri per STFPM single-bee (basato su main_stfpm.py).

Ogni run addestra lo student e ogni --eval-every step calcola sul test set le
mappe teacher/student PER LAYER (alla loro risoluzione nativa). Sulle stesse
mappe vengono poi valutate varianti di post-processing senza riaddestrare
(stesse di sweep_paste.py):
    layer    sottoinsieme dei layer usato per la mappa
    agg      sum | prod (STFPM originale) | nsum (somma di mappe normalizzate
             con media/std globali per layer, stimate su immagini di TRAIN) |
             psum (come nsum ma con media/std per posizione)
    sigma    blur gaussiano (in pixel a 128x128)
    score    max | topk<N> (media degli N pixel più alti, a 128x128)
Le metriche sono img AUROC / img AP e pixel AP a 128x128 (calcolata su GPU su
tutti i pixel del test set).

    CUDA_VISIBLE_DEVICES=0 python experiments/single_bee/sweep_stfpm.py --tag r18_512 --img-size 512
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_stfpm as ms  # noqa: E402
from sweep_paste import EVAL_RES, augment, compute_stats, variant_table  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.stfpm.stfpm import STFPM
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--layers", default="1,2,3", help="layer del backbone (layer<N>) su cui si addestra lo student")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--steps", type=int, default=0, help="step totali (0 = --epochs epoche)")
    p.add_argument("--eval-every", type=int, default=0, help="step tra le valutazioni (0 = ogni epoca)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--sched", default="cos", help="cos | const")
    p.add_argument("--aug", default="", help="flip | rot (flip + rot90)")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--n-stats", type=int, default=1024, help="immagini di train per le statistiche nsum/psum")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def layer_maps(model, x, amp):
    """Mappe (B,1,h,w) di ||t/|t| - s/|s|||^2 per ogni layer."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(amp)):
        t_feat = model.teacher(x)
        s_feat = model.student(x)
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
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        maps = [m.half().cpu() for m in layer_maps(model, images, amp)]
        per_layer = [[m] for m in maps] if per_layer is None else [a + [m] for a, m in zip(per_layer, maps)]
        if with_labels:
            labels.append(np.asarray(batch[1]).reshape(-1))
            m = F.interpolate(batch[2].float(), size=(EVAL_RES, EVAL_RES), mode="area")
            masks.append(m > 0.5)
    per_layer = [torch.cat(a) for a in per_layer]
    if not with_labels:
        return per_layer
    return per_layer, np.concatenate(labels), torch.cat(masks)


def main():
    args = parse_args()
    ms.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    size = (args.img_size, args.img_size)
    ms.IMG_SIZE = size
    layer_ids = [int(x) for x in args.layers.split(",")]
    layers = [f"layer{i}" for i in layer_ids]

    log_dir = ms.REPO_ROOT / "logs" / "stfpm_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VarroaDataset(ms.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(ms.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=12, pin_memory=True)
    stats_idx = np.random.RandomState(0).choice(len(train_ds), args.n_stats, replace=False)
    stats_loader = DataLoader(Subset(train_ds, stats_idx), batch_size=64, num_workers=12)

    teacher = CustomFeatureExtractor(args.backbone, layers, device, frozen=True)
    student = CustomFeatureExtractor(args.backbone, layers, device, frozen=False)
    model = STFPM(teacher, student).to(device)
    teacher.model.to(memory_format=torch.channels_last)
    student.model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    total = args.steps or int(args.epochs * len(train_loader))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
                 if args.sched == "cos" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0))
    eval_every = args.eval_every or len(train_loader)

    config = vars(args)
    history = []
    print(f"[{args.tag}] {config}\n{len(train_loader)} step/epoca, {total} step totali, "
          f"eval ogni {eval_every}", flush=True)

    step, t0, tot, n = 0, time.time(), 0.0, 0
    while step < total:
        for x in tqdm(train_loader, desc=f"step {step}", leave=False):
            model.train()
            x = augment(x.to(device, non_blocking=True), args.aug).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                with torch.no_grad():
                    t_feat = model.teacher(x)
                s_feat = model.student(x)
            loss = 0.0
            for t, s in zip(t_feat, s_feat):
                t = F.normalize(t.float(), dim=1)
                s = F.normalize(s.float(), dim=1)
                loss = loss + (0.5 / (t.shape[2] * t.shape[3])) * F.mse_loss(t, s, reduction="sum")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot += loss.item(); n += 1; step += 1

            if step % eval_every == 0 or step == total:
                lr = scheduler.get_last_lr()[0]
                print(f"[{args.tag}] step {step}/{total} (ep {step / len(train_loader):.2f}) "
                      f"loss {tot / n:.3f} lr {lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
                mean_loss, tot, n = tot / n, 0.0, 0
                t0 = time.time()
                stats = compute_stats(infer(model, stats_loader, device, args.amp, with_labels=False))
                per_layer, labels, masks = infer(model, test_loader, device, args.amp)
                rows = variant_table(per_layer, labels, masks, layer_ids, stats, device)
                del per_layer
                for key in ("img_auc", "pxl_ap"):
                    rows.sort(key=lambda r: -r[key])
                    print(f"  top varianti ({key}):")
                    for r in rows[:5]:
                        print(f"    layers={r['layers']} agg={r['agg']} sigma={r['sigma']} score={r['score']}: "
                              f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap={r['pxl_ap']:.4f}")
                print(f"  eval {time.time() - t0:.0f}s", flush=True)
                history.append({"step": step, "epoch": step / len(train_loader), "lr": lr,
                                "loss": mean_loss, "variants": rows})
                with open(log_dir / f"{args.tag}.json", "w") as f:
                    json.dump({"config": config, "history": history}, f)
                t0 = time.time()
            if step >= total:
                break

    print(f"\n[{args.tag}] FINE", flush=True)


if __name__ == "__main__":
    main()
