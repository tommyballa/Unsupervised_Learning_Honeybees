"""
Costruisce un dataset train/test per l'anomaly detection a partire dai crop
delle singole api, inserendo la varroa sintetica direttamente sui crop
(varroa_on_crops.py).

  1. Divide a caso i crop in train (80%) e test (20%). I crop possono venire da
     più cartelle (es. i cluster di un export dell'hub): la cartella di origine
     viene riportata nella colonna "cluster" dei CSV.
  2. Nel test, il 3% delle api (--anomalous-fraction) riceve la varroa: 2/3 con
     una varroa, 1/3 con due.
  3. Per ogni ape con varroa salva anche la GT mask (bianco = varroa).

Struttura dell'output:
  <output>/train/normal/        api normali di train
  <output>/test/normal/         api normali di test
  <output>/test/anomalous/      api di test con varroa
  <output>/test/masks/          GT mask delle api con varroa (<bee_id>_mask.png, 0/255)
  <output>/test/labels.csv      bee_id, cluster, label (0/1), num_varroa, image_path, mask_path
  <output>/split.csv            ogni crop sorgente con cluster, split, label e num_varroa
  <output>/config.json          parametri e conteggi della generazione

Le api normali sono copie identiche dei crop sorgente (o hardlink/symlink con
--mode). Con lo stesso seed e gli stessi crop il risultato è identico.

Uso:
  python build_varroa_split.py                 # dataset completo
  python build_varroa_split.py --dry-run       # solo i conteggi, non scrive nulla
  python build_varroa_split.py --limit 2000 --output /tmp/prova   # prova su un sottoinsieme
  python build_varroa_split.py --crops-dir <export>/Giovane <export>/Adulte \
      --anomalous-fraction 0.05 --mode hardlink --output <dir>    # più cartelle sorgente
"""

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

# --- Configurazione ---
CROPS_DIR = voc.CROPS_DIR
OUTPUT_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee_batch3+4_varroa_split"

SEED = 42
TEST_FRACTION = 0.20           # quota di api nel test set
ANOMALOUS_FRACTION = 0.03      # quota del test set con varroa
TWO_VARROA_FRACTION = 1 / 3    # quota delle api con varroa che ne ha due (le altre una)

COPY_WORKERS = 16


def plan_counts(n_total, test_fraction=TEST_FRACTION, anomalous_fraction=ANOMALOUS_FRACTION,
                two_fraction=TWO_VARROA_FRACTION):
    """Numero di api per gruppo: (train, test, con una varroa, con due varroa)."""
    n_test = round(n_total * test_fraction)
    n_anomalous = round(n_test * anomalous_fraction)
    n_two = round(n_anomalous * two_fraction)
    return n_total - n_test, n_test, n_anomalous - n_two, n_two


def split_files(files, seed, test_fraction=TEST_FRACTION):
    """Divisione casuale e riproducibile in (train, test); test è in ordine casuale."""
    shuffled = sorted(files)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    return sorted(shuffled[n_test:]), shuffled[:n_test]


def seed_for(path, seed):
    """Seed proprio di ogni ape: il risultato non dipende dall'ordine di elaborazione."""
    return (zlib.crc32(os.path.basename(path).encode()) + seed) % (2 ** 32)


def make_anomalous(path, num_varroa, varroa_sources, seed):
    """Inserisce num_varroa varroa sul crop. Restituisce (immagine, gt_mask) o (None, None)."""
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
    """Copia (o collega) i file sorgente nella destinazione, in parallelo."""
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
    """Crop .png di tutte le cartelle: {path: nome della cartella di origine}.

    Due crop con lo stesso nome in cartelle diverse finirebbero sullo stesso file
    di output, quindi sono un errore.
    """
    clusters, seen = {}, {}
    for d in crops_dirs:
        name = os.path.basename(os.path.normpath(d))
        files = glob.glob(os.path.join(d, "*.png"))
        if not files:
            raise FileNotFoundError(f"Nessun crop .png in {d}")
        for path in files:
            base = os.path.basename(path)
            if base in seen:
                raise ValueError(f"Crop con lo stesso nome in due cartelle: {seen[base]} e {path}")
            seen[base] = path
            clusters[path] = name
    return clusters


def check_output(dirs, expected):
    """Conta i file scritti in ogni cartella: il disco NTFS può perderli senza errori."""
    for name, n in expected.items():
        found = len(os.listdir(dirs[name]))
        if found != n:
            raise RuntimeError(f"{name}: {found} file invece di {n}")


def build(crops_dirs, output_dir, seed=SEED, limit=None, mode="copy", dry_run=False,
          anomalous_fraction=ANOMALOUS_FRACTION):
    if isinstance(crops_dirs, str):
        crops_dirs = [crops_dirs]
    cluster_of = collect_crops(crops_dirs)
    files = sorted(cluster_of)
    if limit is not None:
        files = random.Random(seed).sample(files, min(limit, len(files)))

    n_train, n_test, n_one, n_two = plan_counts(len(files), anomalous_fraction=anomalous_fraction)
    print(f"Crop totali: {len(files)}")
    for name in sorted(set(cluster_of.values())):
        print(f"  {name}: {sum(1 for p in files if cluster_of[p] == name)}")
    print(f"  train: {n_train}")
    print(f"  test:  {n_test}  (con varroa: {n_one + n_two} = {n_one} con una + {n_two} con due)")
    if dry_run:
        return None

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"La cartella di output esiste e non è vuota: {output_dir}")

    dirs = {name: os.path.join(output_dir, *name.split("/"))
            for name in ("train/normal", "test/normal", "test/anomalous", "test/masks")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    train_files, test_files = split_files(files, seed, test_fraction=TEST_FRACTION)

    # --- Api con varroa: si scorre il test in ordine casuale finché le quote
    # sono piene. Se su un'ape la varroa non entra, quell'ape resta normale.
    varroa_sources = vg.load_varroa_sources()
    targets = [1] * n_one + [2] * n_two
    random.Random(seed + 1).shuffle(targets)

    anomalous = {}   # path sorgente -> numero di varroa
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
        raise RuntimeError(f"Inserite solo {len(anomalous)} api con varroa su {len(targets)} richieste")

    # --- Api normali: copie identiche dei crop
    test_normal = [p for p in test_files if p not in anomalous]
    copy_all([(p, os.path.join(dirs["train/normal"], os.path.basename(p))) for p in train_files], mode)
    copy_all([(p, os.path.join(dirs["test/normal"], os.path.basename(p))) for p in test_normal], mode)
    check_output(dirs, {"train/normal": len(train_files), "test/normal": len(test_normal),
                        "test/anomalous": len(anomalous), "test/masks": len(anomalous)})

    # --- Label del test e riepilogo della divisione
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

    print(f"\nApi su cui la varroa non entrava (rimaste normali): {skipped}")
    print(f"Dataset salvato in: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops-dir", nargs="+", default=[CROPS_DIR],
                        help="una o più cartelle di crop (es. i cluster di un export)")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="usa solo N crop a caso (per le prove)")
    parser.add_argument("--anomalous-fraction", type=float, default=ANOMALOUS_FRACTION,
                        help="quota del test set con varroa")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy",
                        help="come portare le api normali nell'output")
    parser.add_argument("--dry-run", action="store_true", help="stampa solo i conteggi")
    args = parser.parse_args()
    build(args.crops_dir, args.output, seed=args.seed, limit=args.limit, mode=args.mode, dry_run=args.dry_run,
          anomalous_fraction=args.anomalous_fraction)


if __name__ == "__main__":
    main()
