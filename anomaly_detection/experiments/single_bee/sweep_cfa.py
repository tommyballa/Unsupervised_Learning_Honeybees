"""
Hyperparameter sweep for single-bee CFA (based on main_cfa.py).

Each run builds the memory bank (reused by runs with the same
configuration), trains for a few hundred steps, evaluates every
--eval-every steps and saves the best checkpoint in checkpoints/single_bee/cfa_sweep/<tag>/.
The metrics are the fast ones (image-level + pixel AUROC/AP at 128x128); with
--variants, each evaluation also prints the metrics of post-processing
variants computed on the same distance maps (blur sigma,
score = max / top-k mean), to choose it without retraining.
The defaults are the paper's CFA; the main_cfa.py configuration is:

    CUDA_VISIBLE_DEVICES=2 python experiments/single_bee/sweep_cfa.py --tag c_l23_768 \
        --bank kmeans --fg --bank-size 32768 --K 1 --layers 2,3 --img-size 768 \
        --lr 1e-5 --epochs 0.36 --eval-every 87 --eval-zero

Summary of the runs: logs/cfa_sweep/summary.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_cfa as mc  # noqa: E402

from moviad.datasets.varroa.varroa_dataset import VarroaDataset
from moviad.models.cfa.cfa import CFA, CFATrainArgs
from moviad.utilities.configurations import Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluation.checkpoints import BestTracker
from moviad.utilities.evaluation.report import print_metrics


LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024}  # wide_resnet50_2
SIGMAS = (0.0, 2.0, 4.0, 8.0)
SCORES = ("max", "topk10", "topk100", "topk1000")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--layers", default="1,2", help="backbone layers, e.g. 1,2 or 1,2,3")
    p.add_argument("--gamma-c", type=int, default=4)
    p.add_argument("--gamma-d", type=int, default=2)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--J", type=int, default=3)
    p.add_argument("--bank", default="mean", help="mean (per-position mean, paper) | kmeans (sampled patches)")
    p.add_argument("--ppi", type=int, default=64, help="patches sampled per image with --bank kmeans")
    p.add_argument("--bank-size", type=int, default=0, help="centroids with --bank kmeans (0 = patches per image / gamma_c)")
    p.add_argument("--fg", action="store_true", help="bank, loss and map only on bee patches (not on the black background)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--sched", default="const", help="const | cos (warmup 2%% + cosine)")
    p.add_argument("--aug", default="none", help="none | flip | flip_rot (flip + 90° rotations)")
    p.add_argument("--eval-every", type=int, default=0, help="steps between evaluations (0 = every epoch)")
    p.add_argument("--variants", action="store_true", help="also print the post-processing variant table at each evaluation")
    p.add_argument("--eval-zero", action="store_true", help="also evaluate before training (random descriptor)")
    # post-processing of the map used to select the best checkpoint (sigma and top-k in pixels at img-size)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--score", default="max", help="max | topk<N> (mean of the N highest pixels)")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def blur(m, sigma):
    if sigma <= 0:
        return m
    return CFA.gaussian_smooth_torch(m.squeeze(1), sigma=sigma).unsqueeze(1)


def image_score(m, mode):
    flat = m.flatten(1)
    if mode == "max":
        return flat.max(1).values
    return flat.topk(int(mode[4:]), dim=1).values.mean(1)


@torch.no_grad()
def infer(model, loader, device):
    """Distance maps (CPU, feature resolution), labels and GT masks at 128x128."""
    model.eval()
    maps, labels, masks = [], [], []
    for images, y, m, _ in tqdm(loader, desc="Inference", leave=False):
        x = images.to(device, non_blocking=True)
        maps.append(model.distance_map(model.describe(x), model.patch_foreground(x)).cpu())
        labels.append(np.asarray(y).reshape(-1))
        masks.append(F.interpolate(m.float(), size=(128, 128), mode="nearest").numpy() > 0.5)
    return torch.cat(maps), np.concatenate(labels), np.concatenate(masks)


def evaluate(dist_maps, size, sigma, device, modes=("max",)):
    """Maps at 128x128 and scores (for each mode) for one post-processing variant."""
    maps, scores = [], {m: [] for m in modes}
    for i in range(0, dist_maps.shape[0], 256):
        mp = F.interpolate(dist_maps[i:i + 256].to(device), size=size, mode="bilinear", align_corners=False)
        mp = blur(mp, sigma)
        for mode in modes:
            scores[mode].append(image_score(mp, mode).cpu().numpy())
        maps.append(F.interpolate(mp, size=(128, 128), mode="bilinear", align_corners=False).cpu().numpy())
    return np.concatenate(maps), {m: np.concatenate(v) for m, v in scores.items()}


def fast_metrics(maps128, scores, labels, masks128):
    """Image-level metrics + pixel AUROC/AP at 128x128 (subsampled to ~4M pixels)."""
    step = max(1, masks128.size // 4_000_000)
    gt, pr = masks128.ravel()[::step], maps128.ravel()[::step]
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12))
    return {"img_roc_auc": roc_auc_score(labels, scores), "img_avg_prec": average_precision_score(labels, scores),
            "img_f1": float(f1), "pxl_roc_auc": roc_auc_score(gt, pr), "pxl_avg_prec": average_precision_score(gt, pr)}


def variant_table(dist_maps, labels, masks128, size, device):
    """Image AUROC / AP and pixel AP (at 128) for several post-processing variants."""
    step = max(1, masks128.size // 2_000_000)
    gt = masks128.ravel()[::step]
    rows = []
    for sigma in SIGMAS:
        maps, scores = evaluate(dist_maps, size, sigma, device, SCORES)
        pxl_ap = average_precision_score(gt, maps.ravel()[::step])
        for mode, sc in scores.items():
            rows.append({"sigma": sigma, "score": mode,
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
    mc.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda:0")
    size = (args.img_size, args.img_size)
    mc.IMG_SIZE = size
    layers = [f"layer{l}" for l in args.layers.split(",")]

    log_dir = mc.REPO_ROOT / "logs" / "cfa_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = mc.REPO_ROOT / "checkpoints" / "single_bee" / "cfa_sweep" / args.tag

    train_ds = VarroaDataset(mc.dataset_args(), "varroa", Split.TRAIN)
    test_ds = VarroaDataset(mc.dataset_args(), "varroa", Split.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                              num_workers=12, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    feature_extractor = CustomFeatureExtractor(mc.BACKBONE, layers, device, frozen=True)
    model = CFA(feature_extractor=feature_extractor, backbone=mc.BACKBONE,
                gamma_c=args.gamma_c, gamma_d=args.gamma_d,
                feature_maps_channels=sum(LAYER_CHANNELS[l] for l in layers),
                K=args.K, J=args.J, bank_init=args.bank, bank_patches_per_image=args.ppi,
                bank_size=args.bank_size or None, foreground_fn=mc.bee_foreground if args.fg else None)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    training_args = CFATrainArgs(epochs=1, batch_size=args.bs, optimizer=optimizer)
    training_args.init_train(model)
    total = int(args.epochs * len(train_loader))
    if args.sched == "cos":
        warm = max(1, int(0.02 * total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    else:
        scheduler = None
    eval_every = args.eval_every or len(train_loader)

    config = {**vars(args), "img_size": list(size)}
    tracker = BestTracker(ckpt_dir, config)
    history = []
    print(f"[{args.tag}] {config}\n{len(train_loader)} steps/epoch, {total} total steps, eval every {eval_every}")

    # the memory bank depends only on these arguments (and on the initial descriptor, saved with it):
    # runs with the same configuration reuse it
    key = "_".join(str(v) for v in (args.img_size, args.layers.replace(",", ""), args.gamma_c, args.gamma_d,
                                    args.bank, args.ppi, args.bank_size, int(args.fg), args.seed))
    cache = ckpt_dir.parent / "_bank_cache" / f"{key}.pth"
    t0 = time.time()
    model.train()
    if cache.exists():
        state = torch.load(cache, map_location=device)
        model.Descriptor.load_state_dict(state["descriptor"])
        model.memory_bank = torch.nn.Parameter(state["memory_bank"], requires_grad=False)
        model.describe(next(iter(train_loader))[:1].to(device))  # sets model.scale
    else:
        model.memory_bank = torch.nn.Parameter(model.initialize_memory_bank(train_loader), requires_grad=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"descriptor": model.Descriptor.state_dict(), "memory_bank": model.memory_bank.data}, cache)
    print(f"[{args.tag}] memory bank {tuple(model.memory_bank.shape)} ({time.time() - t0:.0f}s)", flush=True)

    def run_eval(step, loss, t_train):
        lr = optimizer.param_groups[0]["lr"]
        print(f"[{args.tag}] step {step}/{total} (ep {step / len(train_loader):.2f}) "
              f"loss {loss:.4f} r {model.r.item():.4f} lr {lr:.2e} ({t_train:.0f}s)")
        t0 = time.time()
        dist_maps, labels, masks128 = infer(model, test_loader, device)
        maps, scores = evaluate(dist_maps, size, args.sigma, device, (args.score,))
        results = fast_metrics(maps, scores[args.score], labels, masks128)
        del maps
        print_metrics(results)
        rows = []
        if args.variants:
            rows = variant_table(dist_maps, labels, masks128, size, device)
            rows.sort(key=lambda r: -r["img_auc"])
            print("  top variants (img_auc):")
            for r in rows[:5]:
                print(f"    sigma={r['sigma']} score={r['score']}: "
                      f"auc={r['img_auc']:.4f} ap={r['img_ap']:.4f} pxl_ap128={r['pxl_ap128']:.4f}")
            best_px = max(rows, key=lambda r: r["pxl_ap128"])
            print(f"    best pxl: sigma={best_px['sigma']} pxl_ap128={best_px['pxl_ap128']:.4f}")
        print(f"  eval {time.time() - t0:.0f}s", flush=True)
        tracker.update(step, results, lambda: mc.model_state(model))
        history.append({"step": step, "epoch": step / len(train_loader), "lr": lr, "loss": loss,
                        "metrics": results, "variants": rows})
        with open(log_dir / f"{args.tag}.json", "w") as f:
            json.dump({"config": config, "history": history}, f, indent=1)

    if args.eval_zero:
        run_eval(0, float("nan"), 0.0)

    step, t0, tot, n = 0, time.time(), 0.0, 0
    done = total == 0
    while not done:
        for x in tqdm(train_loader, desc=f"step {step}", leave=False):
            tot += model.train_step(augment(x.to(device, non_blocking=True), args.aug), training_args)
            if scheduler is not None:
                scheduler.step()
            n += 1; step += 1

            if step % eval_every == 0 or step == total:
                run_eval(step, tot / n, time.time() - t0)
                tot, n, t0 = 0.0, 0, time.time()
            if step >= total:
                done = True
                break

    print(f"\n[{args.tag}] BEST — {tracker.summary()}")


if __name__ == "__main__":
    main()
