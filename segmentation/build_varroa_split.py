"""
Builds a train/test dataset for anomaly detection from the crops of the
single bees, inserting the synthetic varroa directly onto the crops
(varroa_on_crops.py).

  1. Randomly splits the crops into train (80%) and test (20%). The crops can come
     from several folders (e.g. the clusters of a hub export): the source folder
     is reported in the "cluster" column of the CSVs.
  2. In the test set, 3% of the bees (--anomalous-fraction) receive varroa: 2/3 with
     one varroa, 1/3 with two.
  3. For each bee with varroa it also saves the GT mask (white = varroa).

Output structure:
  <output>/train/normal/        normal train bees
  <output>/test/normal/         normal test bees
  <output>/test/anomalous/      test bees with varroa
  <output>/test/masks/          GT masks of the bees with varroa (<bee_id>_mask.png, 0/255)
  <output>/test/labels.csv      bee_id, cluster, label (0/1), num_varroa, image_path, mask_path
  <output>/split.csv            each source crop with cluster, split, label and num_varroa
  <output>/config.json          generation parameters and counts

Normal bees are identical copies of the source crops (or hardlinks/symlinks with
--mode). With the same seed and the same crops the result is identical.

Usage:
  python build_varroa_split.py                 # full dataset
  python build_varroa_split.py --dry-run       # counts only, writes nothing
  python build_varroa_split.py --limit 2000 --output /tmp/prova   # trial on a subset
  python build_varroa_split.py --crops-dir <export>/Giovane <export>/Adulte \
      --anomalous-fraction 0.05 --mode hardlink --output <dir>    # several source folders
"""

from bee_paths import DATA_ROOT
import argparse
import csv
import glob
import json
import os
import random
import shutil
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import varroa_generator as vg
import varroa_on_crops as voc

# --- Configuration ---
CROPS_DIR = voc.CROPS_DIR
OUTPUT_DIR = DATA_ROOT + "/processed/single_bee_batch3+4_varroa_split"

SEED = 42
TEST_FRACTION = 0.20           # fraction of bees in the test set
ANOMALOUS_FRACTION = 0.03      # fraction of the test set with varroa
TWO_VARROA_FRACTION = 1 / 3    # fraction of the bees with varroa that have two (the others have one)

COPY_WORKERS = 16


def plan_counts(n_total, test_fraction=TEST_FRACTION, anomalous_fraction=ANOMALOUS_FRACTION,
                two_fraction=TWO_VARROA_FRACTION):
    """Number of bees per group: (train, test, with one varroa, with two varroa)."""
    n_test = round(n_total * test_fraction)
    n_anomalous = round(n_test * anomalous_fraction)
    n_two = round(n_anomalous * two_fraction)
    return n_total - n_test, n_test, n_anomalous - n_two, n_two


def split_files(files, seed, test_fraction=TEST_FRACTION):
    """Random, reproducible split into (train, test); test is in random order."""
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    return sorted(shuffled[n_test:]), shuffled[:n_test]


def seed_for(path, seed):
    """Per-bee seed: the result does not depend on the processing order."""
    return (zlib.crc32(os.path.basename(path).encode()) + seed) % (2 ** 32)


def make_anomalous(path, num_varroa, varroa_sources, seed):
    """Inserts num_varroa varroa on the crop. Returns (image, gt_mask) or (None, None)."""
    crop = cv2.imread(path)
    if crop is None:
        return None, None
    s = seed_for(path, seed)
    random.seed(s)
    np.random.seed(s)
    synthetic, gt_mask, _ = voc.add_varroa_to_crop(
        crop, varroa_sources, (voc.VARROA_BEE_RATIO_MIN, voc.VARROA_BEE_RATIO_MAX), num_varroa=num_varroa
    )
    return synthetic, gt_mask


def copy_all(pairs, mode):
    """Copies (or links) the source files to the destination, in parallel."""
    def one(pair):
        src, dst = pair
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copyfile(src, dst)

    with ThreadPoolExecutor(COPY_WORKERS) as pool:
        list(pool.map(one, pairs))


def collect_crops(crops_dirs):
    """The .png crops of all folders: {path: name of the source folder}.

    Two crops with the same name in different folders would end up in the same output
    file, so they are an error.
    """
    clusters, seen = {}, {}
    for d in crops_dirs:
        name = os.path.basename(os.path.normpath(d))
        files = glob.glob(os.path.join(d, "*.png"))
        if not files:
            raise FileNotFoundError(f"No .png crops in {d}")
        for path in files:
            base = os.path.basename(path)
            if base in seen:
                raise ValueError(f"Crops with the same name in two folders: {seen[base]} and {path}")
            seen[base] = path
            clusters[path] = name
    return clusters


def check_output(dirs, expected):
    """Counts the files written in each folder: the NTFS disk can lose them without errors."""
    for name, n in expected.items():
        found = len(os.listdir(dirs[name]))
        if found != n:
            raise RuntimeError(f"{name}: {found} files instead of {n}")


def build(crops_dirs, output_dir, seed=SEED, limit=None, mode="copy", dry_run=False,
          anomalous_fraction=ANOMALOUS_FRACTION):
    if isinstance(crops_dirs, str):
        crops_dirs = [crops_dirs]
    cluster_of = collect_crops(crops_dirs)
    files = sorted(cluster_of)
    if limit is not None:
        files = random.Random(seed).sample(files, min(limit, len(files)))

    n_train, n_test, n_one, n_two = plan_counts(len(files), anomalous_fraction=anomalous_fraction)
    print(f"Total crops: {len(files)}")
    for name in sorted(set(cluster_of.values())):
        print(f"  {name}: {sum(1 for p in files if cluster_of[p] == name)}")
    print(f"  train: {n_train}")
    print(f"  test:  {n_test}  (with varroa: {n_one + n_two} = {n_one} with one + {n_two} with two)")
    if dry_run:
        return None

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"The output folder exists and is not empty: {output_dir}")

    dirs = {name: os.path.join(output_dir, *name.split("/"))
            for name in ("train/normal", "test/normal", "test/anomalous", "test/masks")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    train_files, test_files = split_files(files, seed, test_fraction=TEST_FRACTION)

    # --- Bees with varroa: the test set is scanned in random order until the quotas
    # are filled. If the varroa does not fit on a bee, that bee stays normal.
    varroa_sources = vg.load_varroa_sources()
    targets = [1] * n_one + [2] * n_two
    random.Random(seed + 1).shuffle(targets)

    anomalous = {}   # source path -> number of varroa
    skipped = 0
    for path in test_files:
        if len(anomalous) == len(targets):
            break
        num_varroa = targets[len(anomalous)]
        synthetic, gt_mask = make_anomalous(path, num_varroa, varroa_sources, seed)
        if synthetic is None:
            skipped += 1
            continue
        bee_id = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(dirs["test/anomalous"], f"{bee_id}.png"), synthetic)
        cv2.imwrite(os.path.join(dirs["test/masks"], f"{bee_id}_mask.png"), gt_mask)
        anomalous[path] = num_varroa
    if len(anomalous) < len(targets):
        raise RuntimeError(f"Only {len(anomalous)} bees with varroa inserted out of {len(targets)} requested")

    # --- Normal bees: identical copies of the crops
    test_normal = [p for p in test_files if p not in anomalous]
    copy_all([(p, os.path.join(dirs["train/normal"], os.path.basename(p))) for p in train_files], mode)
    copy_all([(p, os.path.join(dirs["test/normal"], os.path.basename(p))) for p in test_normal], mode)
    check_output(dirs, {"train/normal": len(train_files), "test/normal": len(test_normal),
                        "test/anomalous": len(anomalous), "test/masks": len(anomalous)})

    # --- Test labels and split summary
    with open(os.path.join(output_dir, "test", "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bee_id", "cluster", "label", "num_varroa", "image_path", "mask_path"])
        for path in sorted(test_files):
            bee_id = os.path.splitext(os.path.basename(path))[0]
            if path in anomalous:
                writer.writerow([bee_id, cluster_of[path], 1, anomalous[path],
                                 os.path.join(dirs["test/anomalous"], f"{bee_id}.png"),
                                 os.path.join(dirs["test/masks"], f"{bee_id}_mask.png")])
            else:
                writer.writerow([bee_id, cluster_of[path], 0, 0,
                                 os.path.join(dirs["test/normal"], f"{bee_id}.png"), ""])

    with open(os.path.join(output_dir, "split.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "cluster", "split", "label", "num_varroa"])
        for path in train_files:
            writer.writerow([os.path.basename(path), cluster_of[path], "train", 0, 0])
        for path in sorted(test_files):
            writer.writerow([os.path.basename(path), cluster_of[path], "test",
                             int(path in anomalous), anomalous.get(path, 0)])

    summary = {
        "crops_dirs": list(crops_dirs),
        "seed": seed,
        "limit": limit,
        "mode": mode,
        "test_fraction": TEST_FRACTION,
        "anomalous_fraction": anomalous_fraction,
        "two_varroa_fraction": TWO_VARROA_FRACTION,
        "varroa_bee_ratio": [voc.VARROA_BEE_RATIO_MIN, voc.VARROA_BEE_RATIO_MAX],
        "counts": {
            "train": len(train_files),
            "test": len(test_files),
            "test_normal": len(test_normal),
            "test_anomalous": len(anomalous),
            "one_varroa": sum(1 for n in anomalous.values() if n == 1),
            "two_varroa": sum(1 for n in anomalous.values() if n == 2),
            "skipped_no_room": skipped,
        },
        "clusters": {
            name: {
                "train": sum(1 for p in train_files if cluster_of[p] == name),
                "test_normal": sum(1 for p in test_normal if cluster_of[p] == name),
                "test_anomalous": sum(1 for p in anomalous if cluster_of[p] == name),
            }
            for name in sorted(set(cluster_of[p] for p in files))
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBees on which the varroa did not fit (left normal): {skipped}")
    print(f"Dataset saved in: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops-dir", nargs="+", default=[CROPS_DIR],
                        help="one or more crop folders (e.g. the clusters of an export)")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="use only N random crops (for trials)")
    parser.add_argument("--anomalous-fraction", type=float, default=ANOMALOUS_FRACTION,
                        help="fraction of the test set with varroa")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="how to bring the normal bees into the output")
    parser.add_argument("--dry-run", action="store_true", help="print the counts only")
    args = parser.parse_args()
    build(args.crops_dir, args.output, seed=args.seed, limit=args.limit, mode=args.mode, dry_run=args.dry_run,
          anomalous_fraction=args.anomalous_fraction)


if __name__ == "__main__":
    main()
