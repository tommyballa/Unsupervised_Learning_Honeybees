"""
Sweep di iperparametri per PatchCore single-bee (basato su main_patchcore.py).

Ogni run costruisce la memory bank con la configurazione data (risoluzione,
layer, patch candidate, dimensione) e la salva in
checkpoints/single_bee/patchcore_sweep/<tag>/ (riusata se esiste già), poi
calcola una sola volta sul test set le distanze dal vicino più prossimo sulla
griglia delle feature e da quelle confronta, senza rifare l'inferenza:
  - i prefissi della memory bank (--bank-sizes): il coreset greedy è annidato,
    i primi k elementi sono il coreset di dimensione k;
  - il sigma del blur della mappa e lo score immagine (max / media top-k pixel
    / "rw5" = media ripesata delle 5 patch peggiori, lo score di main_patchcore).
Le metriche sono quelle veloci (image-level + pixel AUROC/AP a 128x128).

    CUDA_VISIBLE_DEVICES=1 python experiments/single_bee/sweep_patchcore.py --tag l123_512

Log e risultati: logs/patchcore_sweep/<tag>.{log,json}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_patchcore as mp  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.training_args import TrainingArgs
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import load_checkpoint, save_checkpoint


SIGMAS = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0)
SCORES = ("max", "topk10", "topk100", "topk1000")
EVAL_SIZE = (512, 512)      # le mappe si valutano alla risoluzione dei crop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--backbone", default=mp.BACKBONE)
    p.add_argument("--layers", default="1,2,3",
                   help="layer del backbone: 2,3 (layerN di wide_resnet50_2) oppure i nomi, es. features.6,features.13")
    p.add_argument("--train-images", type=int, default=0,
                   help="sottoinsieme casuale del training set per una run veloce (0 = tutte)")
    p.add_argument("--bank-size", type=int, default=30000)
    p.add_argument("--bank-sizes", default="", help="prefissi della bank da valutare, es. 10000,30000 (default: solo --bank-size)")
    p.add_argument("--max-patches", type=int, default=4_000_000, help="patch candidate al coreset")
    p.add_argument("--no-fg", action="store_true", help="niente maschera dell'ape (campionamento, score e mappa)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--test-bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def blur(m, sigma):
    if sigma <= 0:
        return m
    return gaussian_blur(m, kernel_size=2 * int(4.0 * sigma + 0.5) + 1, sigma=sigma)


def image_score(m, mode):
    flat = m.flatten(1)
    if mode == "max":
        return flat.max(1).values
    return flat.topk(int(mode[4:]), dim=1).values.mean(1)


@torch.no_grad()
def infer(patchcore, loader, bank_sizes, use_fg):
    """Distanze NN sulla griglia per ogni prefisso della bank, score rw5, maschere."""
    model = patchcore.vad_model
    device = model.device
    bank = model.memory_bank[0].to(device)
    grids = {k: [] for k in bank_sizes}
    rw5 = {k: [] for k in bank_sizes}
    fgs, labels, masks = [], [], []
    for images, y, m, _ in tqdm(loader, desc="Inferenza", leave=False):
        images = images.to(device, non_blocking=True)
        embedding, b, h, w = model.extract_embedding(images)
        pixel_fg = mp.bee_foreground(images) if use_fg else torch.ones_like(images[:, :1])
        grid_fg = patchcore._grid_mask(images, pixel_fg, (h, w))
        for k in bank_sizes:
            sub = bank[:k]
            scores, locations = model.nearest_neighbors(embedding, n_neighbors=1, memory_bank=sub)
            scores, locations = scores.reshape(b, -1), locations.reshape(b, -1)
            rw5[k].append(model.compute_anomaly_score_old(scores * grid_fg, locations, embedding, sub).cpu().numpy())
            grids[k].append(scores.reshape(b, 1, h, w).half().cpu())
        fgs.append(F.interpolate(pixel_fg, size=EVAL_SIZE, mode="nearest").bool().cpu())
        labels.append(np.asarray(y).reshape(-1))
        masks.append(F.interpolate(m.float(), size=(128, 128), mode="nearest").numpy() > 0.5)
    return ({k: torch.cat(v) for k, v in grids.items()}, {k: np.concatenate(v) for k, v in rw5.items()},
            torch.cat(fgs), np.concatenate(labels), np.concatenate(masks))


def evaluate(grid, fg, sigma, device, modes):
    """Mappe a 128x128 e score (per ogni modo) per una variante di post-processing."""
    maps, scores = [], {m: [] for m in modes}
    for i in range(0, grid.shape[0], 128):
        mp_ = F.interpolate(grid[i:i + 128].to(device).float(), size=EVAL_SIZE, mode="bilinear", align_corners=False)
        mp_ = blur(mp_, sigma) * fg[i:i + 128].to(device)
        for mode in modes:
            scores[mode].append(image_score(mp_, mode).cpu().numpy())
        maps.append(F.interpolate(mp_, size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy())
    return np.concatenate(maps), {m: np.concatenate(v) for m, v in scores.items()}


def image_metrics(labels, scores):
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12))
    return {"img_auc": float(roc_auc_score(labels, scores)),
            "img_ap": float(average_precision_score(labels, scores)), "img_f1": float(f1)}


def main():
    args = parse_args()
    mp.set_seed(args.seed)
    device = torch.device("cuda:0")
    mp.IMG_SIZE = (args.img_size, args.img_size)
    mp.BACKBONE = args.backbone
    mp.LAYERS = [l if not l.isdigit() else f"layer{l}" for l in args.layers.split(",")]
    mp.MEMORY_BANK_SIZE = args.bank_size
    mp.MAX_TRAIN_PATCHES = args.max_patches
    bank_sizes = sorted({int(k) for k in args.bank_sizes.split(",") if k} | {args.bank_size})

    log_dir = mp.REPO_ROOT / "logs" / "patchcore_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mp.REPO_ROOT / "checkpoints" / "single_bee" / "patchcore_sweep" / args.tag
    config = vars(args)
    print(f"[{args.tag}] {config}", flush=True)

    patchcore = mp.build_model(device)
    if args.no_fg:
        patchcore.foreground_fn = None

    t0 = time.time()
    try:
        state, _ = load_checkpoint(ckpt_dir)
        mp.load_model_state(patchcore, state)
        print(f"[{args.tag}] memory bank riusata da {ckpt_dir}")
    except FileNotFoundError:
        train_ds = VarroaDataset(mp.dataset_args(), "varroa", Split.TRAIN)
        if args.train_images and args.train_images < len(train_ds):
            subset = np.random.default_rng(args.seed).choice(len(train_ds), args.train_images, replace=False)
            train_ds = torch.utils.data.Subset(train_ds, subset.tolist())
        training_args = TrainingArgs(batch_size=args.bs, epochs=1)
        patchcore.train()
        patchcore.start_task(0, train_ds, training_args)
        patchcore.train_task(0, train_ds, None, metrics=[], device=device, train_args=training_args)
        save_checkpoint(ckpt_dir, mp.model_state(patchcore), config, best_epoch=None)
    bank = patchcore.vad_model.memory_bank[0]
    print(f"[{args.tag}] memory bank {tuple(bank.shape)} ({time.time() - t0:.0f}s)", flush=True)
    bank_sizes = [k for k in bank_sizes if k <= bank.shape[0]]

    test_ds = VarroaDataset(mp.dataset_args(), "varroa", Split.TEST)
    test_loader = DataLoader(test_ds, batch_size=args.test_bs, shuffle=False, num_workers=12, pin_memory=True)
    t0 = time.time()
    patchcore.eval()
    grids, rw5, fg, labels, masks128 = infer(patchcore, test_loader, bank_sizes, not args.no_fg)
    print(f"[{args.tag}] inferenza {time.time() - t0:.0f}s", flush=True)

    step = max(1, masks128.size // 4_000_000)
    gt = masks128.ravel()[::step]
    rows = []
    for k in bank_sizes:
        rows.append({"bank": k, "sigma": None, "score": "rw5", **image_metrics(labels, rw5[k])})
        for sigma in SIGMAS:
            maps, scores = evaluate(grids[k], fg, sigma, device, SCORES)
            pr = maps.ravel()[::step]
            pxl = {"pxl_auc128": float(roc_auc_score(gt, pr)), "pxl_ap128": float(average_precision_score(gt, pr))}
            for mode, sc in scores.items():
                rows.append({"bank": k, "sigma": sigma, "score": mode, **image_metrics(labels, sc), **pxl})
        print(f"\n[{args.tag}] bank {k}")
        for r in rows:
            if r["bank"] == k:
                pxl = f" pxl_auc128={r['pxl_auc128']:.4f} pxl_ap128={r['pxl_ap128']:.4f}" if "pxl_ap128" in r else ""
                print(f"  sigma={r['sigma']} score={r['score']}: auc={r['img_auc']:.4f} "
                      f"ap={r['img_ap']:.4f} f1={r['img_f1']:.4f}{pxl}", flush=True)
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": config, "variants": rows}, f, indent=1)

    best = max(rows, key=lambda r: r["img_auc"])
    print(f"\n[{args.tag}] BEST img_auc — {best}")


if __name__ == "__main__":
    main()
