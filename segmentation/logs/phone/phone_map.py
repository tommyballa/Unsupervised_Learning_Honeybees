"""The phone bees on the phase 2 UMAP map and their clusters, compared with batches 3-6 and batch 7.

Reads coords.csv from scripts/embedding_map.py (where the phone bees show up as batch_6, because the photo
folder is inside batch_6) and recognises them by the ``phone__`` prefix. In <map>/solo_phone/:
- cluster_per_gruppo.csv: share of each cluster for batches 3-6, batch 7, phone;
- map_holdout*.png: the phone bees (b7 excluded) on top of the grey map of batches 3-6;
- crops_<cluster>.png: up to 24 phone crops per cluster;
- crops_fase1_<stato>.png: phone crops kept / discarded / unassigned by the phase 1 filter.
Usage: python logs/phone/phone_map.py <map folder>
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from bee_paths import AL_WORKDIR, DATA_ROOT  # noqa: E402
from embedding_map import OTHER, SET_ASIDE, UNASSIGNED, english, plot  # noqa: E402

PHONE_DIR = Path(DATA_ROOT + "/processed/single_bee_phone_SAM3/cleaned_bees")
FASE1_PRED = Path(AL_WORKDIR + "/results_sam3_b3456/added_phone_predictions.csv")


def group_of(row):
    if row["image_id"].startswith("phone__"):
        return "phone"
    return "batch_7" if row["batch"] == "batch_7" else "batch_3-6"


def crop_grid(files, out, title, n=24, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    files = list(files)
    if not files:
        return
    rng = np.random.default_rng(seed)
    pick = [files[i] for i in rng.choice(len(files), size=min(n, len(files)), replace=False)]
    cols = 8
    rows = -(-len(pick) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8 + 0.4), dpi=110, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, pick):
        im = Image.open(f).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        ax.imshow(Image.alpha_composite(bg, im))
    fig.suptitle(f"{title} ({len(files)} bees, {len(pick)} at random)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    map_dir = Path(sys.argv[1])
    out_dir = map_dir / "solo_phone"
    out_dir.mkdir(exist_ok=True)
    with open(map_dir / "coords.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = [group_of(r) for r in rows]

    # 1. Cluster shares per group
    names = ["batch_3-6", "batch_7", "phone"]
    clusters = sorted(set(r["cluster"] for r in rows))
    count = {g: {c: 0 for c in clusters} for g in names}
    conf = {g: [] for g in names}
    for r, g in zip(rows, groups):
        count[g][r["cluster"]] += 1
        if r["cluster"] not in (UNASSIGNED, SET_ASIDE):
            conf[g].append(float(r["confidence"]))
    with open(out_dir / "cluster_per_gruppo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster"] + [f"{g}_n" for g in names] + [f"{g}_quota" for g in names])
        for c in clusters:
            w.writerow([c] + [count[g][c] for g in names] +
                       [f"{count[g][c] / max(sum(count[g].values()), 1):.4f}" for g in names])
        w.writerow(["confidenza_mediana"] + [""] * len(names) + [f"{np.median(conf[g]):.4f}" for g in names])
    print(f"{'cluster':20s}" + "".join(f"{g:>16s}" for g in names))
    for c in clusters:
        print(f"{c:20s}" + "".join(f"{count[g][c]:7d} ({100 * count[g][c] / sum(count[g].values()):5.1f}%)" for g in names))
    print(f"{'median confidence':20s}" + "".join(f"{np.median(conf[g]):16.3f}" for g in names))

    # 2. Map: batches 3-6 in grey, phone in colour (batch 7 excluded)
    keep = [i for i, g in enumerate(groups) if g != "batch_7"]
    coords = np.array([[float(rows[i]["x"]), float(rows[i]["y"])] for i in keep])
    cl = [rows[i]["cluster"] for i in keep]
    holdout = np.array([groups[i] == "phone" for i in keep])
    classes = sorted(set(cl) - {UNASSIGNED, SET_ASIDE, OTHER})
    plot(coords, cl, classes, holdout, out_dir, "fase2_sam3 - phone photos")
    (out_dir / "map_overview.png").unlink()       # same as the full map without batch 7: not needed
    (out_dir / "map_by_cluster.png").unlink()

    # 3. Crops per cluster
    for c in clusters:
        files = [PHONE_DIR / r["image_id"].split("__", 1)[1] for r, g in zip(rows, groups) if g == "phone" and r["cluster"] == c]
        crop_grid(files, out_dir / f"crops_{english(c).replace(' ', '_')}.png", f"Phone bees, phase-2 cluster {english(c)}")

    # 4. Crops by outcome of the phase 1 filter
    with open(FASE1_PRED, newline="", encoding="utf-8") as f:
        pred = list(csv.DictReader(f))
    for name, test in (("kept_Ok", lambda p: p["kept"] == "1"),
                       ("Discard", lambda p: p["status"] == "predicted" and p["kept"] == "0"),
                       ("unassigned", lambda p: p["status"] == "unassigned")):
        files = [PHONE_DIR / p["image_id"].split("__", 1)[1] for p in pred if test(p)]
        crop_grid(files, out_dir / f"crops_fase1_{name}.png", f"Phone bees, phase-1 filter: {name.replace('_', ' ')}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
