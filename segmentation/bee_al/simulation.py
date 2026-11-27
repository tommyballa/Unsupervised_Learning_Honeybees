"""Simulation of the hub with an oracle, and learning curves."""
from __future__ import annotations

import copy
import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from .config import ALConfig
from .engine import predict_all
from .knn import nearest_labeled_similarity
from .memory_bank import MemoryBank
from .propagation import LabelSpreader
from .query import select_batch
from .validation import classification_metrics, clustering_metrics


def make_synthetic(
    n: int = 3000,
    dim: int = 64,
    n_classes: int = 6,
    imbalance: float = 0.6,
    noise: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """L2-normalised high-dimensional Gaussians, with imbalanced classes.

    Class ``i`` has a fraction of examples proportional to ``imbalance**i``.
    """
    rng = np.random.default_rng(seed)
    weights = imbalance ** np.arange(n_classes)
    counts = np.maximum(1, np.round(n * weights / weights.sum()).astype(int))
    counts[0] += n - counts.sum()  # the rounding remainder goes to the largest class
    centers = rng.normal(size=(n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = np.concatenate(
        [centers[c] + noise * rng.normal(size=(m, dim)) / np.sqrt(dim) for c, m in enumerate(counts)]
    )
    y = [f"Class_{c}" for c, m in enumerate(counts) for _ in range(m)]
    perm = rng.permutation(len(X))
    X = X[perm].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, [y[i] for i in perm]


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Variant:
    name: str
    title: str
    implemented: bool = True


VARIANTS: dict[str, Variant] = {
    "baseline": Variant("baseline", "Baseline: query random, kNN"),
    "v1": Variant("v1", "V1: kNN uncertainty"),
    "v2": Variant("v2", "V2: uncertainty + diversity, label propagation"),
    "v3": Variant("v3", "V3: extractor contrastive, pseudo-label + consistency", implemented=False),
}
METRICS = ("accuracy", "macro_f1", "nmi", "ari")


def variant_config(cfg: ALConfig, variant: str) -> ALConfig:
    """Config of the variant: same frozen extractor, only how labels are chosen and used changes."""
    out = copy.deepcopy(cfg)
    if variant == "baseline":      # random query, no semi-supervision
        out.query.random_fraction = 1.0
        out.propagation.enabled = False
    elif variant == "v1":          # top-N by kNN uncertainty, no diversity, novelty or propagation
        out.query.candidate_multiplier = 1
        out.query.novelty_fraction = 0.0
        out.query.uncertainty_source = "knn"
        out.propagation.enabled = False
    elif variant == "v2":          # like the hub: uncertainty + diversity, with label propagation
        out.query.uncertainty_source = "propagation"
        out.propagation.enabled = True
    elif variant == "v3":
        raise NotImplementedError(
            "V3 (contrastive extractor + pseudo-label + consistency, re-train every N) is scaffolded "
            "but not implemented: it needs a real FeatureExtractorTrainer, see bee_al/retrain.py"
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return out


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def run_single(
    X: np.ndarray,
    oracle: Sequence[str],
    variant: str,
    cfg: ALConfig,
    seed: int,
    spreader: Optional[LabelSpreader] = None,
) -> list[dict[str, Any]]:
    """Simulate the hub with an oracle answering in place of the annotator.

    ``oracle[i]`` is the label of ``X[i]``: any set of labels works.
    Returns one curve point per round: metrics as a function of the number of labels.
    """
    vcfg = variant_config(cfg, variant)
    n = len(X)
    oracle = list(oracle)
    rng = np.random.default_rng([seed, 0])
    val_idx = np.sort(rng.choice(n, size=min(cfg.validation.size, n // 4), replace=False))
    val_true = [oracle[i] for i in val_idx]

    bank = MemoryBank([str(i) for i in range(n)], X, backend=cfg.knn.backend)
    if vcfg.propagation.enabled and spreader is None:
        p = vcfg.propagation
        spreader = LabelSpreader.from_bank(bank, p.n_neighbors, p.alpha, p.max_iter, p.tol)

    rows: list[dict[str, Any]] = []
    budget = min(cfg.simulation.max_labels, n - len(val_idx))
    while True:
        knn, final = predict_all(bank, spreader, vcfg)
        labeled = bank.labeled_indices()
        if len(labeled):
            predicted = final.labels
            rest = np.setdiff1d(np.arange(n), labeled)  # clusters are evaluated on the unlabelled bees
            row = {"variant": variant, "seed": seed, "n_labels": int(len(labeled)), "n_classes": len(bank.classes)}
            row.update(classification_metrics(val_true, [predicted[i] for i in val_idx]))
            row.update(clustering_metrics([oracle[i] for i in rest], [predicted[i] for i in rest]))
            rows.append(row)
        if len(labeled) >= budget:
            break

        excluded = set(labeled.tolist()) | set(val_idx.tolist()) | set(bank.unknown_indices().tolist())
        source = knn if vcfg.query.uncertainty_source == "knn" else final
        qcfg = dataclasses.replace(vcfg.query, batch_size=min(vcfg.query.batch_size, budget - len(labeled)))
        batch = select_batch(
            X, excluded, source.probs if source.classes else None,
            bank.n_labeled, len(bank.classes), qcfg, np.random.default_rng([seed, len(rows) + 1]),
            novelty=1.0 - nearest_labeled_similarity(bank) if bank.n_labeled else None,
        )
        if not batch:
            break
        for q in batch:
            bank.add_label(bank.ids[q.index], oracle[q.index])
    return rows


def labels_to_threshold(rows: Sequence[dict[str, Any]], metric: str, threshold: float) -> Optional[int]:
    """Minimum number of labels at which the curve reaches the threshold (``None`` if never)."""
    for row in sorted(rows, key=lambda r: r["n_labels"]):
        if row[metric] is not None and row[metric] >= threshold:
            return int(row["n_labels"])
    return None


def run_simulation(
    X: np.ndarray,
    oracle: Sequence[str],
    cfg: ALConfig,
    variants: Sequence[str] = ("baseline", "v1", "v2"),
    seeds: Optional[Sequence[int]] = None,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """All variants over several seeds. Returns ``{"rows": [...], "summary": {...}}``."""
    seeds = list(cfg.simulation.seeds if seeds is None else seeds)
    sim = cfg.simulation
    spreader: Optional[LabelSpreader] = None  # the graph depends only on the embeddings: one for all runs
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "threshold_metric": sim.threshold_metric, "threshold": sim.threshold,
        "seeds": seeds, "n_samples": int(len(X)), "n_classes": len(set(oracle)), "variants": {},
    }

    for name in variants:
        variant = VARIANTS[name]
        if not variant.implemented:
            log(f"{name}: scaffolded but not implemented, skipped")
            summary["variants"][name] = {"title": variant.title, "implemented": False}
            continue
        vcfg = variant_config(cfg, name)
        if vcfg.propagation.enabled and spreader is None:
            p = vcfg.propagation
            bank = MemoryBank([str(i) for i in range(len(X))], X, backend=cfg.knn.backend)
            spreader = LabelSpreader.from_bank(bank, p.n_neighbors, p.alpha, p.max_iter, p.tol)

        per_seed = [run_single(X, oracle, name, cfg, seed, spreader) for seed in seeds]
        rows += [r for run in per_seed for r in run]
        curve = []
        for n_labels in sorted({r["n_labels"] for run in per_seed for r in run}):
            point: dict[str, Any] = {"n_labels": n_labels}
            for metric in METRICS:
                values = [r[metric] for run in per_seed for r in run if r["n_labels"] == n_labels and r[metric] is not None]
                point[f"{metric}_mean"] = float(np.mean(values)) if values else None
                point[f"{metric}_std"] = float(np.std(values)) if values else None
            curve.append(point)
        reached = [labels_to_threshold(run, sim.threshold_metric, sim.threshold) for run in per_seed]
        hit = [v for v in reached if v is not None]
        summary["variants"][name] = {
            "title": variant.title,
            "implemented": True,
            "curve": curve,
            "labels_to_threshold": {
                "per_seed": reached,
                "reached": f"{len(hit)}/{len(seeds)}",
                "mean": float(np.mean(hit)) if hit else None,
                "std": float(np.std(hit)) if hit else None,
            },
        }
        last = curve[-1]
        log(f"{name}: final {sim.threshold_metric} {last[sim.threshold_metric + '_mean']:.3f} "
            f"with {last['n_labels']} labels; threshold {sim.threshold} reached in {len(hit)}/{len(seeds)} seeds"
            + (f" with {np.mean(hit):.0f} ± {np.std(hit):.0f} labels" if hit else ""))
    return {"rows": rows, "summary": summary}


def oracle_subset(ids: Sequence[str], X: np.ndarray, labels: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Restrict the embeddings to the bees the oracle can label."""
    keep = [i for i, image_id in enumerate(ids) if image_id in labels]
    return X[keep], [labels[ids[i]] for i in keep]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

# First three hues of the categorical palette: also validated for colour blindness on every pair.
_SERIES_COLORS = {"baseline": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a"}
_SURFACE, _INK, _INK_MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_METRIC_TITLES = {
    "accuracy": "Accuracy (validation set)", "macro_f1": "Macro-F1 (validation set)",
    "nmi": "Cluster NMI", "ari": "Cluster ARI",
}


def plot_learning_curves(summary: dict[str, Any], path: Union[str, Path]) -> None:
    """Learning curves: one panel per metric, mean ± standard deviation over the seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = {k: v for k, v in summary["variants"].items() if v.get("implemented")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, facecolor=_SURFACE)
    for ax, metric in zip(axes.ravel(), METRICS):
        ax.set_facecolor(_SURFACE)
        for name, data in variants.items():
            pts = [p for p in data["curve"] if p[f"{metric}_mean"] is not None]
            x = np.array([p["n_labels"] for p in pts])
            mean = np.array([p[f"{metric}_mean"] for p in pts])
            std = np.array([p[f"{metric}_std"] for p in pts])
            color = _SERIES_COLORS.get(name, _INK_MUTED)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    label=data["title"])
        if metric == summary["threshold_metric"]:
            ax.axhline(summary["threshold"], color=_INK_MUTED, linewidth=1)
            ax.annotate(f"threshold {summary['threshold']:g}", xy=(0.995, summary["threshold"]),
                        xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color=_INK_MUTED)
        ax.set_title(_METRIC_TITLES[metric], loc="left", fontsize=11, color=_INK)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=_GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)
    for ax in axes[1]:
        ax.set_xlabel("Number of labels", fontsize=10, color=_INK_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10, labelcolor=_INK)
    fig.suptitle(
        f"Learning curves: mean ± std. dev. over {len(summary['seeds'])} seeds "
        f"({summary['n_samples']} bees, {summary['n_classes']} classes)",
        x=0.01, ha="left", fontsize=12, color=_INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def save_results(result: dict[str, Any], out_dir: Union[str, Path], plot: bool = True) -> dict[str, Path]:
    """Write ``results.csv`` (one point per variant/seed/round), ``summary.json`` and the PNG plot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / "results.csv", "json": out / "summary.json"}
    fields = ["variant", "seed", "n_labels", "n_classes", *METRICS]
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)
    if plot and any(v.get("implemented") for v in result["summary"]["variants"].values()):
        paths["png"] = out / "learning_curves.png"
        plot_learning_curves(result["summary"], paths["png"])
    return paths
