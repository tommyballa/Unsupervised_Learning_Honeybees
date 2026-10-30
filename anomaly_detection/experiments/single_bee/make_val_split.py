"""
Divide il test set del dataset finale SAM3 in validation e test held-out.

La selezione della best epoca/step (e la soglia dell'F1) si fa sul validation
set; il test held-out si valuta una sola volta sul checkpoint scelto.

    <DATASET_DIR>/val/            VAL_ANOMALOUS_FRACTION delle anomale + VAL_NORMALS sane
    <DATASET_DIR>/test_heldout/   le altre anomale + le stesse 5000 sane del test
                                  ridotto usato finora (VarroaDataset, seed 0)

Le anomale sono divise stratificando per cluster e numero di varroa; le sane
di validation sono prese (stratificate per cluster) tra quelle fuori dal test
ridotto. Stessa struttura di test/ (labels.csv, normal/, anomalous/, masks/),
file come hardlink. Con lo stesso seed il risultato è identico.

    python experiments/single_bee/make_val_split.py [--dry-run]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path("/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3_varroa_split")
VAL_ANOMALOUS_FRACTION = 0.25
VAL_NORMALS = 1250            # 20% di anomale come nel test ridotto
SEED = 0
# devono coincidere con VarroaDataset per ritrovare le stesse 5000 sane del test
MAX_TEST_NORMALS = 5000
TEST_SUBSAMPLE_SEED = 0


def stratified_take(df, keys, fraction, rng):
    """Indici di circa `fraction` righe per ogni gruppo (almeno 1 se il gruppo ne ha 2+)."""
    taken = []
    for _, g in df.groupby(keys):
        n = int(round(len(g) * fraction))
        if n == 0 and len(g) >= 2:
            n = 1
        taken.extend(rng.choice(g.index, size=n, replace=False))
    return pd.Index(sorted(taken))


def write_split(df, src_dir, out_dir, dry_run):
    print(f"{out_dir.name}: {len(df)} ({(df.label == 0).sum()} sane, {(df.label == 1).sum()} con varroa)")
    print(pd.crosstab(df.cluster, [df.label, df.num_varroa]).to_string(), "\n")
    if dry_run:
        return
    for sub in ("normal", "anomalous", "masks"):
        (out_dir / sub).mkdir(parents=True, exist_ok=False)
    rows = []
    for r in df.itertuples():
        sub = "normal" if r.label == 0 else "anomalous"
        img = out_dir / sub / f"{r.bee_id}.png"
        os.link(src_dir / sub / f"{r.bee_id}.png", img)
        mask = ""
        if r.label == 1:
            mask = out_dir / "masks" / f"{r.bee_id}_mask.png"
            os.link(src_dir / "masks" / f"{r.bee_id}_mask.png", mask)
        rows.append([r.bee_id, r.cluster, r.label, r.num_varroa, str(img), str(mask)])
    pd.DataFrame(rows, columns=["bee_id", "cluster", "label", "num_varroa", "image_path", "mask_path"]) \
        .to_csv(out_dir / "labels.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="solo i conteggi, non scrive nulla")
    args = parser.parse_args()

    src = DATASET_DIR / "test"
    df = pd.read_csv(src / "labels.csv")

    # le 5000 sane del test ridotto, con la stessa estrazione di VarroaDataset
    normal_idx = df.index[df.label == 0]
    rng = np.random.default_rng(TEST_SUBSAMPLE_SEED)
    dropped = rng.choice(normal_idx, size=len(normal_idx) - MAX_TEST_NORMALS, replace=False)
    test_normals = normal_idx.difference(dropped)

    rng = np.random.default_rng(SEED)
    anomalous = df[df.label == 1]
    val_anom = stratified_take(anomalous, ["cluster", "num_varroa"], VAL_ANOMALOUS_FRACTION, rng)
    pool = df.loc[dropped]
    val_norm = stratified_take(pool, ["cluster"], VAL_NORMALS / len(pool), rng)

    val = df.loc[val_norm.union(val_anom)]
    test = df.loc[test_normals.union(anomalous.index.difference(val_anom))]
    assert not set(val.bee_id) & set(test.bee_id)

    write_split(val, src, DATASET_DIR / "val", args.dry_run)
    write_split(test, src, DATASET_DIR / "test_heldout", args.dry_run)


if __name__ == "__main__":
    main()
