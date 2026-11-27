"""
Heatmaps of the hard cases of the held-out test: the images that most of the
8 models misclassify at the image threshold chosen on the validation set.

Image scores come from logs/score_dist/<model>.npz (score_distributions.py),
thresholds and heatmap normalisation from heatmaps/single_bee_val/<model>/results.json.
For each class, the --per-class images misclassified by the largest number
of models are taken (ties broken by the mean score percentile furthest "on the
wrong side"); then each model reruns inference only on those images.

Output in heatmaps/hard_cases/:
    hard_cases.csv                     index, class, path, models that get it wrong, scores
    grid_varroa.png / grid_healthy.png rows = images, columns = original, GT,
                                       overlay of each model (✗ = wrong)
    <model>/...png                   full 5-panel figure (as in plot_heatmaps.py)

    CUDA_VISIBLE_DEVICES=2 python experiments/single_bee/hard_cases.py
"""

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "single_bee"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_heatmaps import load_best  # noqa: E402
from score_distributions import MODELS, LABELS  # noqa: E402

REPO_ROOT = EXPERIMENTS.parent
SCORE_DIR = REPO_ROOT / "logs" / "score_dist"
OUT_DIR = REPO_ROOT / "heatmaps" / "hard_cases"


def load_scores():
    scores, thresholds, norms, labels = {}, {}, {}, None
    for name in MODELS:
        npz = np.load(SCORE_DIR / f"{name}.npz")
        results = json.loads((REPO_ROOT / "heatmaps" / "single_bee_val" / name / "results.json").read_text())
        if labels is not None and not np.array_equal(labels, npz["labels"]):
            raise ValueError(f"{name}: test set order differs from the other models")
        labels = npz["labels"]
        scores[name] = npz["scores"]
        thresholds[name] = results["image_threshold"]
        norms[name] = results["heatmap_normalization"]
    return scores, thresholds, norms, labels


def select_hard(scores, thresholds, labels, per_class):
    wrong = np.stack([(scores[m] >= thresholds[m]) != (labels == 1) for m in MODELS])
    # score percentile of each model: high = "looks like varroa"
    pct = np.stack([rankdata(scores[m]) / len(labels) for m in MODELS]).mean(0)
    selected = {}
    for label in (1, 0):
        idx = np.flatnonzero(labels == label)
        # varroa: low percentile = harder; healthy: high percentile
        wrong_side = pct[idx] if label == 1 else -pct[idx]
        order = np.lexsort((wrong_side, -wrong[:, idx].sum(0)))
        selected[label] = idx[order[:per_class]]
    return selected, wrong


@torch.no_grad()
def model_maps(name, indices, device):
    from moviad.datasets.varroa.varroa_dataset import VarroaDataset
    from moviad.utilities.configurations import Split
    from moviad.utilities.evaluation.report import run_inference

    mod = importlib.import_module(f"main_{name}")
    mod.set_seed(mod.SEED)
    model, _ = load_best(mod, device)
    dataset = VarroaDataset(mod.dataset_args(), "varroa", Split.TEST)
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=getattr(mod, "TEST_BATCH_SIZE", mod.BATCH_SIZE), shuffle=False,
                        num_workers=mod.NUM_WORKERS, pin_memory=True)
    maps, scores, _, masks = run_inference(model, loader, device, getattr(mod, "map_postprocess", None))
    del model
    torch.cuda.empty_cache()
    return dataset, maps, scores, masks, getattr(mod, "RAW_LABEL", "Raw anomaly map")


def save_grid(rows, title, save_path):
    from moviad.utilities.evaluation.report import make_overlay, normalize_anomaly_map

    ncols = 2 + len(MODELS)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.9 * ncols, 2.05 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["img"])
        axes[r, 0].set_title(f"#{row['idx']}  ({row['n_wrong']}/{len(MODELS)} wrong)", fontsize=8)
        axes[r, 1].imshow(row["mask"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title("Ground truth", fontsize=8)
        for c, name in enumerate(MODELS, start=2):
            m = row["models"][name]
            axes[r, c].imshow(make_overlay(row["img"], normalize_anomaly_map(m["map"], m["norm"])))
            if row["mask"].any():
                axes[r, c].contour(row["mask"], levels=[0.5], colors="white", linewidths=0.7)
            mark = "✗" if m["wrong"] else "✓"
            axes[r, c].set_title(f"{LABELS[name]} {mark}\n{m['score']:.3g} (thr {m['thr']:.3g})",
                                 fontsize=7, color="#c0392b" if m["wrong"] else "#1f1e1b")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    from moviad.utilities.evaluation.report import load_rgb, save_heatmap_figure

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=10, help="hard cases per class")
    args = parser.parse_args()

    scores, thresholds, norms, labels = load_scores()
    selected, wrong = select_hard(scores, thresholds, labels, args.per_class)
    indices = np.concatenate([selected[1], selected[0]])
    for label, text in [(1, "Varroa (false negatives)"), (0, "Healthy (false positives)")]:
        print(f"{text}: errors per image {wrong[:, selected[label]].sum(0).tolist()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    per_image = {int(i): {"models": {}} for i in indices}
    image_paths = None
    for name in MODELS:
        dataset, maps, batch_scores, masks, raw_label = model_maps(name, indices, device)
        image_paths = dataset.image_paths
        (OUT_DIR / name).mkdir(exist_ok=True)
        for k, i in enumerate(indices):
            i = int(i)
            # check: the score recomputed on the subset must match the cached one
            if not np.isclose(batch_scores[k], scores[name][i], rtol=1e-3, atol=1e-5):
                print(f"  WARNING {name} #{i}: score {batch_scores[k]:.5g} vs cache {scores[name][i]:.5g}")
            entry = per_image[i]
            if "img" not in entry:
                entry["img"] = load_rgb(image_paths[i], masks.shape[-2:])
                entry["mask"] = masks[k, 0]
            is_wrong = bool(wrong[MODELS.index(name), i])
            entry["models"][name] = {"map": maps[k, 0], "norm": norms[name], "score": float(batch_scores[k]),
                                     "thr": thresholds[name], "wrong": is_wrong}
            gt_name = "Varroa" if labels[i] == 1 else "Healthy"
            pred = "Varroa" if batch_scores[k] >= thresholds[name] else "Healthy"
            title = (f"{LABELS[name]} | GT: {gt_name} | Pred: {pred} | score {batch_scores[k]:.4f} "
                     f"(threshold {thresholds[name]:.4f}) | {int(wrong[:, i].sum())}/{len(MODELS)} models wrong")
            save_heatmap_figure(entry["img"], masks[k, 0], maps[k, 0], norms[name], title,
                                OUT_DIR / name / f"{gt_name}_{i:05d}.png", raw_label=raw_label)

    for label, fname, title in [(1, "grid_varroa.png", "Varroa missed by most models"),
                                (0, "grid_healthy.png", "Healthy bees flagged by most models")]:
        rows = [{**per_image[int(i)], "idx": int(i), "n_wrong": int(wrong[:, i].sum())} for i in selected[label]]
        save_grid(rows, title, OUT_DIR / fname)

    with open(OUT_DIR / "hard_cases.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "image_path", "n_wrong", "wrong_models"]
                        + [f"score_{m}" for m in MODELS])
        for i in indices:
            writer.writerow([int(i), "varroa" if labels[i] == 1 else "healthy", image_paths[i],
                             int(wrong[:, i].sum()),
                             " ".join(m for k, m in enumerate(MODELS) if wrong[k, i])]
                            + [f"{scores[m][i]:.5g}" for m in MODELS])
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
