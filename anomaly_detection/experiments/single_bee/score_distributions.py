"""
Distribution of image scores on the held-out test, one panel per model.

Reloads the official best checkpoint of each model (checkpoints/single_bee_val/,
same path as plot_heatmaps.py), computes the image scores on the held-out
test and saves them in logs/score_dist/<model>.npz (reused if they exist:
--recompute to redo them). The F1 threshold drawn is the one chosen on the
validation set (image_threshold in heatmaps/single_bee_val/<model>/results.json).
Figures: logs/score_dist/score_distributions.{pdf,png} (grid) and\nscore_distribution_<model>.{pdf,png} (one per model).

    CUDA_VISIBLE_DEVICES=1 python experiments/single_bee/score_distributions.py
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "single_bee"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_heatmaps import load_best  # noqa: E402

REPO_ROOT = EXPERIMENTS.parent
OUT_DIR = REPO_ROOT / "logs" / "score_dist"
MODELS = ["patchcore", "cfa", "dinomaly", "padim", "stfpm", "fastflow", "paste", "rd4ad"]
LABELS = {"patchcore": "PatchCore", "cfa": "CFA", "dinomaly": "Dinomaly", "padim": "PaDiM",
          "stfpm": "STFPM", "fastflow": "FastFlow", "paste": "PaSTe", "rd4ad": "RD4AD"}
NORMAL_COLOR, VARROA_COLOR = "#2a78d6", "#e34948"


@torch.no_grad()
def test_scores(name, device):
    from moviad.datasets.varroa.varroa_dataset import VarroaDataset
    from moviad.utilities.configurations import Split

    mod = importlib.import_module(f"main_{name}")
    mod.set_seed(mod.SEED)
    model, _ = load_best(mod, device)
    model.eval()
    loader = DataLoader(VarroaDataset(mod.dataset_args(), "varroa", Split.TEST),
                        batch_size=getattr(mod, "TEST_BATCH_SIZE", mod.BATCH_SIZE), shuffle=False,
                        num_workers=mod.NUM_WORKERS, pin_memory=True)
    scores, labels = [], []
    for images, batch_labels, _, _ in tqdm(loader, desc=name):
        _, batch_scores = model(images.to(device))
        scores.append(batch_scores.float().cpu().numpy().reshape(-1))
        labels.append(np.asarray(batch_labels).reshape(-1))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(scores), np.concatenate(labels)


def draw_panel(ax, name, scores, labels, threshold):
    bins = np.linspace(scores.min(), scores.max(), 60)
    for label, color, text in [(0, NORMAL_COLOR, "Healthy"), (1, VARROA_COLOR, "Varroa")]:
        ax.hist(scores[labels == label], bins=bins, density=True, color=color, alpha=0.55,
                edgecolor="white", linewidth=0.4, label=text)
    if threshold is not None:
        ax.axvline(threshold, color="#1f1e1b", linestyle="--", linewidth=1.2, label="F1 threshold (val)")
    ax.set_title(f"{LABELS[name]}   AUROC {roc_auc_score(labels, scores):.3f}", fontsize=10,
                 loc="left", color="#1f1e1b")
    ax.set_yticks([])
    ax.tick_params(colors="#55544e", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#8a8980")


def plot(data):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, name in zip(axes.flat, MODELS):
        if name not in data:
            ax.set_visible(False)
            continue
        draw_panel(ax, name, *data[name])
    for ax in axes[1]:
        ax.set_xlabel("image anomaly score", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=9)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=3, frameon=False, fontsize=9, loc="lower center")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "score_distributions.pdf")
    fig.savefig(OUT_DIR / "score_distributions.png", dpi=150)
    plt.close(fig)

    # one figure per model (to include them separately)
    for name in data:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        draw_panel(ax, name, *data[name])
        ax.set_xlabel("image anomaly score", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"score_distribution_{name}.pdf")
        fig.savefig(OUT_DIR / f"score_distribution_{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    data = {}
    for name in MODELS:
        cache = OUT_DIR / f"{name}.npz"
        if args.recompute or not cache.exists():
            scores, labels = test_scores(name, device)
            np.savez(cache, scores=scores, labels=labels)
        npz = np.load(cache)
        results = json.loads((REPO_ROOT / "heatmaps" / "single_bee_val" / name / "results.json").read_text())
        data[name] = (npz["scores"], npz["labels"], results.get("image_threshold"))
        auroc = roc_auc_score(npz["labels"], npz["scores"])
        print(f"{name:>10s} AUROC {auroc:.4f} (results.json {results['metrics']['img_roc_auc']:.4f})")
    plot(data)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
