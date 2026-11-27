"""Congela uno snapshot delle predizioni senza avviare l'hub.

Uso:
    python scripts/export_snapshot.py --name 2026-10-05_round42

Scrive in ``<results_dir>/snapshots/<name>/``: predictions.csv (una riga per ape, con frame di
origine), frames.csv, meta.json e una copia di al_labels.csv e validation_set.json.
Non aggiunge round ad al_rounds.jsonl. Se l'hub è acceso, lo snapshot riflette le etichette
salvate su disco in questo momento.
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
    ap.add_argument("--name", required=True, help="nome dello snapshot (lettere, cifre, '_', '-', '.')")
    ap.add_argument("--config", default=None, help="file di configurazione (default config/al_config.yaml)")
    ap.add_argument("--annotator", default=None, help="chi congela lo snapshot (default: utente di sistema)")
    args = ap.parse_args()

    engine = ALEngine.from_config(args.config, persist_rounds=False)
    try:
        out = engine.export_snapshot(args.name, args.annotator or getpass.getuser())
    except ALError as e:
        sys.exit(f"Snapshot non creato: {e}")
    counts = read_meta(out)["status_counts"]
    print(f"Snapshot scritto in {out}")
    print("Api per stato: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
