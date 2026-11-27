"""2D map (UMAP) of the embeddings of a phase, coloured by the cluster assigned by the model.

Usage:
    python scripts/embedding_map.py --config config/al_fase2_sam3.yaml --name fase2_v004

The labels are those the hub would show now: the human label if there is one, otherwise the predicted
class (kNN + propagation); the "unassigned" and the set-aside bees are kept apart. UMAP is fitted
only on the non-held-out bees (on a fixed sample, for speed) and then all the others are projected
onto it: the bees of a held-out batch end up on the same map without having distorted it.

In ``<results_dir>/embedding_map/<name>/``:
- ``coords.csv``: coordinates, batch, cluster and confidence of each bee;
- ``map_overview.png``: all non-held-out bees, one colour per cluster;
- ``map_by_cluster.png``: one panel per cluster (the other bees in grey);
- ``map_holdout.png``: the held-out bees over the map of the others (if there are any);
- ``map_holdout_by_cluster.png``: the same, one panel per cluster;
- with ``--per-batch``: ``map_<batch>.png`` and ``map_<batch>_by_cluster.png`` for each non-held-out batch.

In the plots the cluster names are in English (``ENGLISH``); ``coords.csv`` keeps the hub's names.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.engine import ALEngine  # noqa: E402

# Categorical palette in fixed order (validated: adjacent pairs ok, hence direct labels and per-cluster panels).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNASSIGNED, SET_ASIDE, OTHER = "Unassigned", "Set aside", "Other"
GREY, BACKGROUND, DARK = "#9a9890", "#e3e2dc", "#4a4944"
INK, MUTED = "#1f1e1c", "#6b6a64"
# Cluster names in the plots (colours still follow the fixed order of the original names).
ENGLISH = {"Adulte": "Adult", "Fuco": "Drone", "Giovane": "Young", "Malata": "Sick",
           "Non Classificabile": "Unclassifiable", "Scarta": "Discard", "Ok": "Ok"}


def english(name: str) -> str:
    return ENGLISH.get(name, name)


def cluster_of(engine: ALEngine) -> tuple[list[str], np.ndarray, list[str]]:
    """Cluster of each bee, confidence and source (``human`` / ``predicted`` / ``""``)."""
    bank, pred = engine.bank, engine.final_pred
    labels, confidence = pred.labels, pred.confidence
    unknown = set(bank.unknown_indices().tolist())
    out, conf, source = [], np.zeros(len(bank)), []
    for i, image_id in enumerate(bank.ids):
        human = bank.label_of(image_id)
        if i in unknown:
            out.append(SET_ASIDE), source.append("human")
        elif human:
            out.append(human), source.append("human")
            conf[i] = 1.0
        elif engine.unassigned[i]:
            out.append(UNASSIGNED), source.append("")
        elif labels[i]:
            out.append(labels[i]), source.append("predicted")
            conf[i] = confidence[i]
        else:
            out.append(OTHER), source.append("")
    return out, conf, source


def plot(coords, clusters, classes, holdout, out_dir, title, batches=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    colour = {c: SERIES[k % len(SERIES)] for k, c in enumerate(classes)}
    colour.update({UNASSIGNED: GREY, SET_ASIDE: GREY, OTHER: GREY})
    clusters = np.asarray(clusters, dtype=object)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.edgecolor": BACKGROUND,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    lim = (coords[:, 0].min() - 1, coords[:, 0].max() + 1), (coords[:, 1].min() - 1, coords[:, 1].max() + 1)

    def frame(ax):
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    def size_of(n):
        return 1.0 if n > 20000 else 4.0       # small clusters get bigger points, otherwise they vanish

    def scatter(ax, idx, scale=1.0):
        # From the largest cluster to the smallest: Adulte (almost 9 bees out of 10) does not cover the others.
        groups = sorted(set(clusters[idx]), key=lambda c: -np.sum(clusters[idx] == c))
        for c in groups:
            own = idx[clusters[idx] == c]
            ax.scatter(coords[own, 0], coords[own, 1], s=scale * size_of(len(own)), c=colour[c],
                       alpha=0.35 if len(own) > 20000 else 0.8, linewidths=0, rasterized=True)

    def names(ax, idx):
        # The name goes where the cluster is densest (the median of a sparse cluster falls in empty space).
        for c in classes:
            members = idx[clusters[idx] == c]
            if len(members) >= 20:
                h, xe, ye = np.histogram2d(coords[members, 0], coords[members, 1], bins=12, range=lim)
                bx, by = np.unravel_index(np.argmax(h), h.shape)
                if h[bx, by] < 0.1 * len(members):
                    continue                    # cluster spread over the whole map: no name, the legend is enough
                ax.text((xe[bx] + xe[bx + 1]) / 2, (ye[by] + ye[by + 1]) / 2, english(c), ha="center", va="bottom",
                        fontsize=10, weight="bold", color=INK, path_effects=halo)

    def legend(ax, idx):
        from matplotlib.lines import Line2D

        shown = [c for c in classes + [UNASSIGNED, SET_ASIDE, OTHER] if np.any(clusters[idx] == c)]
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=colour[c], markeredgewidth=0,
                          label=f"{english(c)} ({int(np.sum(clusters[idx] == c))})") for c in shown]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=INK)

    train = np.flatnonzero(~holdout)
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    scatter(ax, train)
    names(ax, train)
    legend(ax, train)
    frame(ax)
    ax.set_title(f"{title}: {len(train)} bees, colour = cluster", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "map_overview.png", bbox_inches="tight")
    plt.close(fig)

    def by_cluster(idx, filename, subtitle):
        """One panel per cluster of the bees ``idx``, over the grey map of the non-held-out bees."""
        panels = [c for c in classes + [UNASSIGNED] if np.any(clusters[idx] == c)]
        cols = 3
        rows = -(-len(panels) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), dpi=130, squeeze=False)
        for ax, c in zip(axes.flat, panels):
            own = idx[clusters[idx] == c]
            ax.scatter(coords[train, 0], coords[train, 1], s=0.5, c=BACKGROUND, linewidths=0, rasterized=True)
            ax.scatter(coords[own, 0], coords[own, 1], s=size_of(len(own)), c=DARK if c == UNASSIGNED else colour[c],
                       alpha=0.35 if len(own) > 20000 else 0.8, linewidths=0, rasterized=True)
            ax.set_title(f"{english(c)} ({len(own)})", loc="left", fontsize=11)
            frame(ax)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{title}: {subtitle}", x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    by_cluster(train, "map_by_cluster.png", "one panel per cluster (other bees in grey)")

    def overlay(idx, stem, what):
        """The bees ``idx`` in colour over the grey map of the non-held-out bees, then one panel per cluster."""
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
        ax.scatter(coords[train, 0], coords[train, 1], s=0.5, c=BACKGROUND, linewidths=0, rasterized=True)
        scatter(ax, idx, scale=2.0)
        names(ax, idx)
        legend(ax, idx)
        frame(ax)
        ax.set_title(f"{title}: {len(idx)} {what} (colour) on the map of the non-held-out bees (grey)", loc="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)
        by_cluster(idx, f"{stem}_by_cluster.png", f"{what}, one panel per cluster (non-held-out bees in grey)")

    held = np.flatnonzero(holdout)
    if len(held):
        overlay(held, "map_holdout", "held-out bees")
    if batches is not None:
        batches = np.asarray(batches, dtype=object)
        for b in sorted(set(batches[train])):
            overlay(train[batches[train] == b], f"map_{b}", f"bees of {b}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config of the phase")
    ap.add_argument("--name", required=True, help="name of the results folder")
    ap.add_argument("--fit-sample", type=int, default=40000, help="(non-held-out) bees on which UMAP is fitted")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--title", default="", help="plot title with --replot")
    ap.add_argument("--replot", action="store_true", help="redraw from the coordinates already saved in coords.csv")
    ap.add_argument("--per-batch", action="store_true", help="also one map per non-held-out batch")
    args = ap.parse_args()
    if args.replot:
        from bee_al.config import load_config

        cfg = load_config(args.config)
        out_dir = cfg.results_dir / "embedding_map" / args.name
        with open(out_dir / "coords.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        coords = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        clusters = [r["cluster"] for r in rows]
        holdout = np.array([r["holdout"] == "1" for r in rows])
        classes = sorted(set(clusters) - {UNASSIGNED, SET_ASIDE, OTHER})
        title = f"{cfg.stage.name or 'stage'} - " + (args.name if not args.title else args.title)
        batches = [r["batch"] for r in rows] if args.per_batch else None
        plot(coords, clusters, classes, holdout, out_dir, title, batches)
        print(f"Done: {out_dir}")
        return

    import umap

    engine = ALEngine.from_config(args.config, persist_rounds=False)
    cfg, bank = engine.cfg, engine.bank
    out_dir = cfg.results_dir / "embedding_map" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters, confidence, source = cluster_of(engine)
    batches = engine.bee_batches()
    holdout = engine._holdout.copy()
    classes = sorted(set(engine.final_pred.classes) | set(bank.classes))
    print(f"Model {engine.model_version}: {len(bank)} bees, {int(holdout.sum())} held out, classes {classes}")

    train = np.flatnonzero(~holdout)
    rng = np.random.default_rng(cfg.seed)
    fit = np.sort(rng.choice(train, size=min(args.fit_sample, len(train)), replace=False))
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="cosine", random_state=cfg.seed)
    coords = np.zeros((len(bank), 2), dtype=np.float32)
    coords[fit] = reducer.fit_transform(bank.X[fit])
    rest = np.setdiff1d(np.arange(len(bank)), fit)
    for start in range(0, len(rest), 20000):
        chunk = rest[start:start + 20000]
        coords[chunk] = reducer.transform(bank.X[chunk])
        print(f"  projected {min(start + 20000, len(rest))}/{len(rest)}", flush=True)

    with open(out_dir / "coords.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "batch", "holdout", "cluster", "source", "confidence", "fit", "x", "y"])
        in_fit = np.zeros(len(bank), dtype=bool)
        in_fit[fit] = True
        for i, image_id in enumerate(bank.ids):
            writer.writerow([image_id, batches[i], int(holdout[i]), clusters[i], source[i], f"{confidence[i]:.4f}",
                             int(in_fit[i]), f"{coords[i, 0]:.4f}", f"{coords[i, 1]:.4f}"])
    title = f"{cfg.stage.name or 'stage'} - model {engine.model_version}"
    plot(coords, clusters, classes, holdout, out_dir, title, list(batches) if args.per_batch else None)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
