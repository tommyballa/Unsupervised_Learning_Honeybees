"""Mappa 2D (UMAP) degli embedding di una fase, colorata con il cluster assegnato dal modello.

Uso:
    python scripts/embedding_map.py --config config/al_fase2_sam3.yaml --name fase2_v004

Le etichette sono quelle che l'hub mostrerebbe ora: etichetta umana se c'è, altrimenti la classe
predetta (kNN + propagazione); a parte le "non assegnate" e le api messe da parte. UMAP si adatta
solo alle api non tenute da parte (su un campione fisso, per la velocità) e poi vi proietta tutte
le altre: le api di un batch tenuto da parte finiscono nella stessa mappa senza averla deformata.

In ``<results_dir>/embedding_map/<name>/``:
- ``coords.csv``: coordinate, batch, cluster e confidenza di ogni ape;
- ``map_overview.png``: tutte le api non tenute da parte, un colore per cluster;
- ``map_by_cluster.png``: un pannello per cluster (le altre api in grigio);
- ``map_holdout.png``: le api tenute da parte sopra la mappa delle altre (se ce ne sono);
- ``map_holdout_by_cluster.png``: le stesse, un pannello per cluster;
- con ``--per-batch``: ``map_<batch>.png`` e ``map_<batch>_by_cluster.png`` per ogni batch non tenuto da parte.

Nei grafici i nomi dei cluster sono in inglese (``ENGLISH``); in ``coords.csv`` restano quelli dell'hub.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.engine import ALEngine  # noqa: E402

# Palette categoriale in ordine fisso (validata: coppie adiacenti ok, quindi etichette dirette e pannelli per cluster).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNASSIGNED, SET_ASIDE, OTHER = "Unassigned", "Set aside", "Other"
GREY, BACKGROUND, DARK = "#9a9890", "#e3e2dc", "#4a4944"
INK, MUTED = "#1f1e1c", "#6b6a64"
# Nomi dei cluster nei grafici (i colori seguono comunque l'ordine fisso dei nomi originali).
ENGLISH = {"Adulte": "Adult", "Fuco": "Drone", "Giovane": "Young", "Malata": "Sick",
           "Non Classificabile": "Unclassifiable", "Scarta": "Discard", "Ok": "Ok"}


def english(name: str) -> str:
    return ENGLISH.get(name, name)


def cluster_of(engine: ALEngine) -> tuple[list[str], np.ndarray, list[str]]:
    """Cluster di ogni ape, confidenza e fonte (``human`` / ``predicted`` / ``""``)."""
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
        return 1.0 if n > 20000 else 4.0       # i cluster piccoli con punti più grandi, altrimenti spariscono

    def scatter(ax, idx, scale=1.0):
        # Dal cluster più grande al più piccolo: Adulte (quasi 9 api su 10) non copre gli altri.
        groups = sorted(set(clusters[idx]), key=lambda c: -np.sum(clusters[idx] == c))
        for c in groups:
            own = idx[clusters[idx] == c]
            ax.scatter(coords[own, 0], coords[own, 1], s=scale * size_of(len(own)), c=colour[c],
                       alpha=0.35 if len(own) > 20000 else 0.8, linewidths=0, rasterized=True)

    def names(ax, idx):
        # Il nome va dove il cluster è più denso (la mediana di un cluster sparso cade nel vuoto).
        for c in classes:
            members = idx[clusters[idx] == c]
            if len(members) >= 20:
                h, xe, ye = np.histogram2d(coords[members, 0], coords[members, 1], bins=12, range=lim)
                bx, by = np.unravel_index(np.argmax(h), h.shape)
                if h[bx, by] < 0.1 * len(members):
                    continue                    # cluster sparso su tutta la mappa: niente nome, basta la legenda
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
    fig.savefig(out_dir / "map_overview.png")
    plt.close(fig)

    def by_cluster(idx, filename, subtitle):
        """Un pannello per cluster delle api ``idx``, sopra la mappa grigia delle api non tenute da parte."""
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
        """Le api ``idx`` a colori sopra la mappa grigia delle api non tenute da parte, poi un pannello per cluster."""
        fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
        ax.scatter(coords[train, 0], coords[train, 1], s=0.5, c=BACKGROUND, linewidths=0, rasterized=True)
        scatter(ax, idx, scale=2.0)
        names(ax, idx)
        legend(ax, idx)
        frame(ax)
        ax.set_title(f"{title}: {len(idx)} {what} (colour) on the map of the non-held-out bees (grey)", loc="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png")
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
    ap.add_argument("--config", required=True, help="config della fase")
    ap.add_argument("--name", required=True, help="nome della cartella dei risultati")
    ap.add_argument("--fit-sample", type=int, default=40000, help="api (non tenute da parte) su cui si adatta UMAP")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--title", default="", help="titolo dei grafici con --replot")
    ap.add_argument("--replot", action="store_true", help="ridisegna dalle coordinate già salvate in coords.csv")
    ap.add_argument("--per-batch", action="store_true", help="anche una mappa per ogni batch non tenuto da parte")
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
        print(f"Fatto: {out_dir}")
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
    print(f"Modello {engine.model_version}: {len(bank)} api, {int(holdout.sum())} tenute da parte, classi {classes}")

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
        print(f"  proiettate {min(start + 20000, len(rest))}/{len(rest)}", flush=True)

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
    print(f"Fatto: {out_dir}")


if __name__ == "__main__":
    main()
