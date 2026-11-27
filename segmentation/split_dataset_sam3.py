"""
Splits the dataset labelled with SAM 3 (generate_labels_sam3.py) into train / val / test
for YOLO training.

The photos are taken directly from SOURCE_DIR (images/ and labels/). The split is random
per single photo and is done day by day (shooting date in the EXIF, one per batch),
so that every split contains every day. Among TRIALS random assignments, the one
closest to the requested fractions, both in photos and in bees, is kept.

Output structure:
  <output>/{train,val,test}/images/   identical copies of the photos
  <output>/{train,val,test}/labels/   identical copies of the labels
  <output>/split.csv                  each photo with day, number of bees and split
  <output>/config.json                parameters and counts

With the same seed and the same labels the result is identical.

Usage:
  python split_dataset_sam3.py --dry-run    # summary only, writes nothing
  python split_dataset_sam3.py              # creates the dataset
"""

from bee_paths import DATA_ROOT
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# --- Configuration ---
BASE_DIR = DATA_ROOT
SOURCE_DIR = os.path.join(BASE_DIR, "processed", "bee_batch3+4+5+6_SAM3")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "bee_batch3+4+5+6_SAM3_split")

SEED = 42
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRIALS = 20000                 # random assignments tried for each day

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
COPY_WORKERS = 16


def shot_day(path):
    """Shooting date (YYYY-MM-DD) from the photo's EXIF."""
    with Image.open(path) as im:
        exif = im.getexif()
        value = exif.get_ifd(0x8769).get(36867) or exif.get(306)   # DateTimeOriginal, otherwise DateTime
    if not value:
        raise ValueError(f"No shooting date in the EXIF of {path}")
    return value[:10].replace(":", "-")


def count_bees(label_path):
    """Number of bees (non-empty lines) in a label file."""
    with open(label_path) as f:
        return sum(1 for line in f if line.strip())


def scan_photos(source_dir):
    """One entry per photo in the dataset; each photo must have its label and vice versa."""
    names = sorted(n for n in os.listdir(os.path.join(source_dir, "images")) if n.lower().endswith(IMAGE_EXTENSIONS))
    labels = {os.path.splitext(n)[0] for n in os.listdir(os.path.join(source_dir, "labels")) if n.endswith(".txt")}
    stems = [os.path.splitext(n)[0] for n in names]
    if len(set(stems)) != len(stems) or set(stems) != labels:
        raise ValueError(f"Photos and labels do not match in {source_dir}: {sorted(set(stems) ^ labels)[:5]} "
                         "(is generate_labels_sam3.py still running?)")
    return [{
        "name": name,
        "day": shot_day(os.path.join(source_dir, "images", name)),
        "n_bees": count_bees(os.path.join(source_dir, "labels", stem + ".txt")),
    } for name, stem in zip(names, stems)]


def assign_photos(n_bees, fractions, rng, trials=TRIALS):
    """Chooses the split of each photo of a day.

    Among `trials` random assignments, keeps the one whose photo and bee fractions are closest
    to `fractions` (chi-squared).
    """
    names = list(fractions)
    target = np.array([fractions[name] for name in names])
    shares = np.column_stack([np.ones(len(n_bees)), np.asarray(n_bees, dtype=float)])
    shares = shares / shares.sum(axis=0)

    choice = rng.choice(len(names), size=(trials, len(shares)), p=target)
    cost = np.zeros(trials)
    for split, fraction in enumerate(target):
        got = (choice == split).astype(float) @ shares          # trials x (photos, bees)
        cost += ((got - fraction) ** 2 / fraction).sum(axis=1)
    return [names[split] for split in choice[cost.argmin()]]


def plan_split(photos, fractions=FRACTIONS, seed=SEED, trials=TRIALS):
    """Adds the split to each photo."""
    rng = np.random.default_rng(seed)
    by_day = defaultdict(list)
    for photo in photos:
        by_day[photo["day"]].append(photo)
    for day in sorted(by_day):
        day_photos = sorted(by_day[day], key=lambda p: p["name"])
        splits = assign_photos([p["n_bees"] for p in day_photos], fractions, rng, trials)
        for photo, split in zip(day_photos, splits):
            photo["split"] = split
    return photos


def summarize(photos, fractions=FRACTIONS):
    """Counts per split: photos, bees, mean and maximum bees per photo."""
    summary = {}
    for split in fractions:
        counts = [p["n_bees"] for p in photos if p["split"] == split]
        summary[split] = {
            "images": len(counts),
            "bees": sum(counts),
            "mean_bees": round(sum(counts) / len(counts), 1) if counts else 0,
            "max_bees": max(counts, default=0),
        }
    return summary


def print_summary(photos, fractions=FRACTIONS):
    summary = summarize(photos, fractions)
    n_images = sum(s["images"] for s in summary.values())
    n_bees = sum(s["bees"] for s in summary.values())
    print(f"{'split':<6}{'photos':>6}{'%':>7}{'bees':>9}{'%':>7}{'mean':>7}{'max':>6}")
    for split, s in summary.items():
        print(f"{split:<6}{s['images']:>6}{100 * s['images'] / n_images:>7.1f}{s['bees']:>9}"
              f"{100 * s['bees'] / n_bees:>7.1f}{s['mean_bees']:>7.0f}{s['max_bees']:>6}")
    print(f"{'total':<6}{n_images:>6}{'':>7}{n_bees:>9}")

    print("\nPhotos per shooting day (" + " / ".join(fractions) + "):")
    for day in sorted({p["day"] for p in photos}):
        counts = [sum(1 for p in photos if (p["day"], p["split"]) == (day, split)) for split in fractions]
        print(f"  {day}  " + " / ".join(f"{c:>3}" for c in counts))


def copy_all(pairs):
    """Copies the files in parallel."""
    with ThreadPoolExecutor(COPY_WORKERS) as pool:
        list(pool.map(lambda pair: shutil.copyfile(*pair), pairs))


def build(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, dry_run=False):
    photos = plan_split(scan_photos(source_dir))
    print_summary(photos)
    if dry_run:
        return photos
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"The output folder exists and is not empty: {output_dir}")

    pairs = []
    for split in FRACTIONS:
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(output_dir, split, kind), exist_ok=True)
    for photo in photos:
        label = os.path.splitext(photo["name"])[0] + ".txt"
        pairs.append((os.path.join(source_dir, "images", photo["name"]),
                      os.path.join(output_dir, photo["split"], "images", photo["name"])))
        pairs.append((os.path.join(source_dir, "labels", label),
                      os.path.join(output_dir, photo["split"], "labels", label)))
    copy_all(pairs)

    with open(os.path.join(output_dir, "split.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "day", "n_bees", "split"])
        for photo in photos:
            writer.writerow([photo["name"], photo["day"], photo["n_bees"], photo["split"]])

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "source_dir": source_dir,
            "seed": SEED,
            "fractions": FRACTIONS,
            "trials": TRIALS,
            "splits": summarize(photos),
        }, f, indent=2)

    print(f"\nDataset created in {output_dir}")
    return photos


def main():
    parser = argparse.ArgumentParser(description="Splits the SAM 3 dataset into train / val / test.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="output folder")
    parser.add_argument("--dry-run", action="store_true", help="print the summary only, write nothing")
    args = parser.parse_args()
    build(output_dir=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
