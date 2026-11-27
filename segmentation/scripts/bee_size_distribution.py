"""Distribuzione della grandezza delle api estratte singolarmente.

Le misure vanno prese su raw_bees (crop RGBA a risoluzione originale, la maschera è nel
canale alpha): in cleaned_bees ogni ape è già riscalata a 512x512 con riempimento fisso,
quindi lì la grandezza è uguale per tutte.

Uso:
    python scripts/bee_size_distribution.py                 # raw_bees accanto a paths.dataset_dir
    python scripts/bee_size_distribution.py --raw-dir /percorso/raw_bees --out data/size_distribution

Per ogni ape (in pixel dell'immagine sorgente):
    area_px     pixel della maschera
    length_px   lato lungo del rettangolo minimo ruotato (lunghezza del corpo, non dipende dall'orientamento)
    width_px    lato corto dello stesso rettangolo
    bbox_w/h    bounding box dritto della maschera (il crop salvato ha 20 px di padding in più)
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
    ("length_px", "Lunghezza (lato lungo, px)"),
    ("width_px", "Larghezza (lato corto, px)"),
    ("area_px", "Area della maschera (px²)"),
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
        # Taglia la coda oltre il 99.5° percentile, altrimenti pochi outlier schiacciano l'istogramma
        hi = np.percentile(values, 99.5)
        ax.hist(values[values <= hi], bins=60, color="#3b6fb6", edgecolor="white", linewidth=0.4)
        median = np.median(values)
        ax.axvline(median, color="#333333", linewidth=1, linestyle="--")
        ax.annotate(f"mediana {median:,.0f}", xy=(median, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color="#333333")
        ax.set_xlabel(label)
        ax.set_ylabel("Numero di api")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="file di configurazione (default config/al_config.yaml)")
    ap.add_argument("--raw-dir", default=None, help="cartella dei crop RGBA (default: raw_bees accanto a paths.dataset_dir)")
    ap.add_argument("--out", default="data/size_distribution", help="cartella di output (CSV + PNG)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="solo le prime N immagini (prova veloce)")
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
    # Un'ape è "tenuta" se dataset_cleaning.py l'ha lasciata in cleaned_bees (stesso nome file)
    df["kept"] = df["filename"].isin({p.name for p in cleaned_dir.glob("*.png")})

    df.to_csv(out_dir / "bee_sizes.csv", index=False)
    cols = [c for c, _ in METRICS] + ["bbox_w", "bbox_h"]
    pcts = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for name, sub in (("tutte (raw_bees)", df), ("tenute (cleaned_bees)", df[df["kept"]])):
        if sub.empty:
            continue
        print(f"\n== {name}: {len(sub)} api ==")
        print(sub[cols].describe(percentiles=pcts).round(1).to_string())

    plot(df, out_dir / "bee_sizes_raw.png", f"Grandezza delle api estratte — tutte ({len(df):,})")
    if df["kept"].any():
        kept = df[df["kept"]]
        plot(kept, out_dir / "bee_sizes_cleaned.png", f"Grandezza delle api estratte — dopo la pulizia ({len(kept):,})")
    print(f"\nScritti in {out_dir}: bee_sizes.csv, bee_sizes_raw.png, bee_sizes_cleaned.png")


if __name__ == "__main__":
    main()
