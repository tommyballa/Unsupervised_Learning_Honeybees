"""Freeze a snapshot of the predictions without starting the hub.

Usage:
    python scripts/export_snapshot.py --name 2026-10-05_round42

Writes to ``<results_dir>/snapshots/<name>/``: predictions.csv (one row per bee, with its source
frame), frames.csv, meta.json and a copy of al_labels.csv and validation_set.json.
It does not add rounds to al_rounds.jsonl. If the hub is running, the snapshot reflects the labels
saved on disk at this moment.
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.engine import ALEngine, ALError  # noqa: E402
from bee_al.snapshot import read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="snapshot name (letters, digits, '_', '-', '.')")
    ap.add_argument("--config", default=None, help="configuration file (default config/al_config.yaml)")
    ap.add_argument("--annotator", default=None, help="who freezes the snapshot (default: system user)")
    args = ap.parse_args()

    engine = ALEngine.from_config(args.config, persist_rounds=False)
    try:
        out = engine.export_snapshot(args.name, args.annotator or getpass.getuser())
    except ALError as e:
        sys.exit(f"Snapshot not created: {e}")
    counts = read_meta(out)["status_counts"]
    print(f"Snapshot written to {out}")
    print("Bees per status: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
