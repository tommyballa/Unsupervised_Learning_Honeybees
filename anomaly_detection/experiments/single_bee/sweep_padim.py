"""
Sweep di iperparametri per PaDiM single-bee (basato su main_padim.py).

Ogni run stima le gaussiane con la configurazione data (backbone, layer,
risoluzione, canali, gaussiana per posizione o condivisa) e le salva in
checkpoints/single_bee/padim_sweep/<tag>/ (riusate se esistono già), poi
calcola una sola volta sul test set le distanze di Mahalanobis sulla griglia
delle feature (una volta per ogni --eps, la regolarizzazione della
covarianza: non serve rifare il fitting) e da quelle confronta, senza rifare l'inferenza, il sigma del
blur della mappa, lo score immagine (max / media top-k pixel) e la maschera
dell'ape (mappa azzerata fuori dall'ape).
Le metriche sono quelle veloci (image-level + pixel AUROC/AP a 128x128).

    CUDA_VISIBLE_DEVICES=1 python experiments/single_bee/sweep_padim.py --tag wrn_l123_256

Log e risultati: logs/padim_sweep/<tag>.{log,json}
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
import main_padim as mp  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.checkpoints import load_checkpoint, save_checkpoint


SIGMAS = (0.0, 4.0, 8.0, 16.0)       # px a 512
SCORES = ("max", "topk1000")
EVAL_SIZE = (512, 512)      # le mappe si valutano alla risoluzione dei crop


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--backbone", default=mp.BACKBONE)
    p.add_argument("--layers", default="1,2,3",
                   help="layer del backbone: 1,2,3 (layerN dei resnet) oppure i nomi, es. features.4,features.7")
    p.add_argument("--proj-dim", type=int, default=mp.PROJ_DIM, help="canali della random projection (0 = tutti)")
    p.add_argument("--gaussian", choices=("position", "shared"), default=mp.GAUSSIAN)
    p.add_argument("--pool", type=int, default=1, help="avg pool locale delle feature")
    p.add_argument("--eps", default="0.01,0.001,0.0001,0.00001", help="regolarizzazioni della covarianza da valutare")
    p.add_argument("--train-images", type=int, default=20000, help="0 = tutte")
    p.add_argument("--bs", type=int, default=32)
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
def infer(model, loader):
    """Distanze sulla griglia, maschere dell'ape a EVAL_SIZE, label e maschere GT a 128."""
    grids, fgs, labels, masks = [], [], [], []
    for images, y, m, _ in tqdm(loader, desc="Inferenza", leave=False):
        images = images.to(model.device, non_blocking=True)
        grids.append(model.distances(model.embed(images)).half().cpu())
        fg = F.interpolate(images, size=EVAL_SIZE, mode="bilinear", align_corners=False)
        fgs.append(mp.bee_foreground(fg).bool().cpu())
        labels.append(np.asarray(y).reshape(-1))
        masks.append(F.interpolate(m.float(), size=(128, 128), mode="nearest").numpy() > 0.5)
    return torch.cat(grids), torch.cat(fgs), np.concatenate(labels), np.concatenate(masks)


def evaluate(grid, fg, sigma, use_fg, device):
    maps, scores = [], {m: [] for m in SCORES}
    for i in range(0, grid.shape[0], 128):
        m = F.interpolate(grid[i:i + 128].to(device).float(), size=EVAL_SIZE, mode="bilinear", align_corners=False)
        m = blur(m, sigma)
        if use_fg:
            m = m * fg[i:i + 128].to(device)
        for mode in SCORES:
            scores[mode].append(image_score(m, mode).cpu().numpy())
        maps.append(F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy())
    return np.concatenate(maps), {k: np.concatenate(v) for k, v in scores.items()}


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
    mp.LAYERS = [f"layer{l}" if l.isdigit() else l for l in args.layers.split(",")]
    mp.PROJ_DIM = args.proj_dim
    mp.GAUSSIAN = args.gaussian
    mp.FEAT_POOL = args.pool
    mp.TRAIN_IMAGES = args.train_images
    mp.SEED = args.seed

    log_dir = mp.REPO_ROOT / "logs" / "padim_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mp.REPO_ROOT / "checkpoints" / "single_bee" / "padim_sweep" / args.tag
    config = vars(args)
    print(f"[{args.tag}] {config}", flush=True)

    model = mp.build_model(device)
    print(f"[{args.tag}] griglia {model.grid}, {len(model.dims)} canali", flush=True)
    t0 = time.time()
    try:
        state, _ = load_checkpoint(ckpt_dir)
        mp.load_model_state(model, state)
        print(f"[{args.tag}] gaussiane riusate da {ckpt_dir}")
    except FileNotFoundError:
        loader = DataLoader(mp.train_dataset(), batch_size=args.bs, shuffle=False, num_workers=12, pin_memory=True)
        model.fit(loader)
        save_checkpoint(ckpt_dir, mp.model_state(model), config, best_epoch=None)
    print(f"[{args.tag}] fitting {time.time() - t0:.0f}s", flush=True)

    test_ds = VarroaDataset(mp.dataset_args(), "varroa", Split.TEST)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=12, pin_memory=True)
    rows = []
    for eps in [float(e) for e in args.eps.split(",")]:
        model.set_precision(eps)
        t0 = time.time()
        grid, fg, labels, masks128 = infer(model, test_loader)
        print(f"\n[{args.tag}] eps {eps}: inferenza {time.time() - t0:.0f}s", flush=True)
        step = max(1, masks128.size // 4_000_000)
        gt = masks128.ravel()[::step]
        for use_fg in (True,):            # senza maschera dell'ape: sempre peggio o uguale
            for sigma in SIGMAS:
                maps, scores = evaluate(grid, fg, sigma, use_fg, device)
                pr = maps.ravel()[::step]
                pxl = {"pxl_auc128": float(roc_auc_score(gt, pr)), "pxl_ap128": float(average_precision_score(gt, pr))}
                for mode, sc in scores.items():
                    r = {"eps": eps, "fg": use_fg, "sigma": sigma, "score": mode, **image_metrics(labels, sc), **pxl}
                    rows.append(r)
                    print(f"  eps={eps} fg={use_fg} sigma={sigma} score={mode}: auc={r['img_auc']:.4f} "
                          f"ap={r['img_ap']:.4f} f1={r['img_f1']:.4f} pxl_auc128={r['pxl_auc128']:.4f} "
                          f"pxl_ap128={r['pxl_ap128']:.4f}", flush=True)
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": config, "variants": rows}, f, indent=1)

    best = max(rows, key=lambda r: r["img_auc"])
    print(f"\n[{args.tag}] BEST img_auc — {best}")


if __name__ == "__main__":
    main()
