"""Size distribution of the individually extracted bees.

Measurements must be taken on raw_bees (RGBA crops at original resolution, the mask is in the
alpha channel): in cleaned_bees every bee is already rescaled to 512x512 with fixed padding,
so there the size is the same for all of them.

Usage:
    python scripts/bee_size_distribution.py                 # raw_bees next to paths.dataset_dir
    python scripts/bee_size_distribution.py --raw-dir /path/raw_bees --out data/size_distribution

For each bee (in pixels of the source image):
    area_px     mask pixels
    length_px   long side of the minimum rotated rectangle (body length, independent of orientation)
    width_px    short side of the same rectangle
    bbox_w/h    axis-aligned bounding box of the mask (the saved crop has 20 px of extra padding)
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import load_config  # noqa: E402

METRICS = [
    ("length_px", "Length (long side, px)"),
    ("width_px", "Width (short side, px)"),
    ("area_px", "Mask area (px²)"),
]


def measure(path: str) -> dict | None:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    mask = (img[:, :, 3] > 0).astype(np.uint8)
    area = int(mask.sum())
    if area == 0:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(largest)
    _, _, bw, bh = cv2.boundingRect(largest)
    return {
        "filename": Path(path).name,
        "area_px": area,
        "length_px": max(rw, rh),
        "width_px": min(rw, rh),
        "bbox_w": bw,
        "bbox_h": bh,
        "crop_w": img.shape[1],
        "crop_h": img.shape[0],
    }


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for ax, (col, label) in zip(axes, METRICS):
        values = df[col].to_numpy()
        # Cut the tail beyond the 99.5th percentile, otherwise a few outliers squash the histogram
        hi = np.percentile(values, 99.5)
        ax.hist(values[values <= hi], bins=60, color="#3b6fb6", edgecolor="white", linewidth=0.4)
        median = np.median(values)
        ax.axvline(median, color="#333333", linewidth=1, linestyle="--")
        ax.annotate(f"median {median:,.0f}", xy=(median, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color="#333333")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of bees")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configuration file (default config/al_config.yaml)")
    ap.add_argument("--raw-dir", default=None, help="folder of the RGBA crops (default: raw_bees next to paths.dataset_dir)")
    ap.add_argument("--out", default="data/size_distribution", help="output folder (CSV + PNG)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cleaned_dir = Path(cfg.dataset_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else cleaned_dir.parent / "raw_bees"
    out_dir = cfg.resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(str(p) for p in raw_dir.glob("*.png"))[: args.limit]
    print(f"{len(paths)} crop in {raw_dir}")
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(measure, paths, chunksize=256) if r is not None]
    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    # A bee is "kept" if dataset_cleaning.py left it in cleaned_bees (same file name)
    df["kept"] = df["filename"].isin({p.name for p in cleaned_dir.glob("*.png")})

    df.to_csv(out_dir / "bee_sizes.csv", index=False)
    cols = [c for c, _ in METRICS] + ["bbox_w", "bbox_h"]
    pcts = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for name, sub in (("all (raw_bees)", df), ("kept (cleaned_bees)", df[df["kept"]])):
        if sub.empty:
            continue
        print(f"\n== {name}: {len(sub)} bees ==")
        print(sub[cols].describe(percentiles=pcts).round(1).to_string())

    plot(df, out_dir / "bee_sizes_raw.png", f"Size of the extracted bees — all ({len(df):,})")
    if df["kept"].any():
        kept = df[df["kept"]]
        plot(kept, out_dir / "bee_sizes_cleaned.png", f"Size of the extracted bees — after cleaning ({len(kept):,})")
    print(f"\nWritten to {out_dir}: bee_sizes.csv, bee_sizes_raw.png, bee_sizes_cleaned.png")


if __name__ == "__main__":
    main()
