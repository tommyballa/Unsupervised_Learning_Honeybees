"""Contrastive fine-tuning of the extractor: process launched by the hub (bee_al/contrastive.py).

Usage:
    python scripts/retrain_contrastive.py --job models/_work/job.json

The job file contains hyperparameters, labels, pseudo-labels and the list of images.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.contrastive import train_contrastive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    with open(args.job, "r", encoding="utf-8") as f:
        job = json.load(f)
    summary = train_contrastive(job, log=lambda s: print(s, flush=True))
    print(f"done in {summary['duration_s']}s, loss {summary['loss_first']:.3f} -> {summary['loss_last']:.3f}", flush=True)


if __name__ == "__main__":
    main()
