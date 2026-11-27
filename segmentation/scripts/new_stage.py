"""Prepare the next phase of the pipeline starting from a snapshot.

Phase 1 separates, for example, well-segmented bees from badly segmented ones. Phase 2
works only on the bees that phase 1 kept: this script lists them and writes a new
config, with labels and models in their own folders, so the two phases do not mix.

Usage:
    python scripts/new_stage.py --snapshot fase1_finale --keep "Segmentata Bene" --name fase2
    BEE_AL_CONFIG=config/al_fase2.yaml ../venv/bin/python -u app.py      # from active_learning_tool/

The new phase gets the snapshot bees whose final class is in --keep, whether labelled by hand
or assigned by the model (--min-confidence discards low-confidence assignments).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.audit import class_normalizer  # noqa: E402
from bee_al.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config  # noqa: E402
from bee_al.snapshot import read_csv, snapshot_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="snapshot of the current phase to start from")
    ap.add_argument("--keep", required=True, nargs="+", help="classes to carry into the new phase")
    ap.add_argument("--name", required=True, help="name of the new phase (letters, digits, '_', '-')")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="minimum confidence for bees assigned by the model")
    ap.add_argument("--config", default=None, help="config of the current phase (default config/al_config.yaml)")
    ap.add_argument("--out-config", default=None, help="config to write (default config/al_<name>.yaml)")
    args = ap.parse_args()

    if not args.name.replace("_", "").replace("-", "").isalnum():
        sys.exit("Invalid stage name: use only letters, digits, '_' and '-'")
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    try:
        folder = snapshot_path(cfg, args.snapshot)
    except ValueError as e:
        sys.exit(str(e))
    if not (folder / "predictions.csv").exists():
        sys.exit(f"Snapshot '{args.snapshot}' not found in {folder.parent}")

    normalize = class_normalizer(cfg)
    keep = {normalize(c) for c in args.keep}
    rows = read_csv(folder / "predictions.csv")
    present = Counter(r["final_label"] for r in rows if r["final_label"])
    missing = keep - set(present)
    if missing:
        sys.exit(f"Classes not present in the snapshot: {sorted(missing)}. Available: {dict(present)}")
    chosen = [r for r in rows if r["final_label"] in keep and r["status"] in ("labeled", "predicted")
              and (r["source"] == "human" or float(r["confidence"]) >= args.min_confidence)]
    if not chosen:
        sys.exit("No bee meets the criteria: nothing to carry into the new phase")

    out_config = Path(args.out_config) if args.out_config else REPO_ROOT / "config" / f"al_{args.name}.yaml"
    results = cfg.results_dir.parent / f"results_{args.name}"
    subset = results / "subset.json"
    for path in (out_config, subset):
        if path.exists():
            sys.exit(f"{path} already exists: choose another name for the stage")
    results.mkdir(parents=True, exist_ok=True)
    with open(subset, "w", encoding="utf-8") as f:
        json.dump({"from_snapshot": args.snapshot, "from_config": str(config_path), "keep": sorted(keep),
                   "min_confidence": args.min_confidence, "image_ids": [r["image_id"] for r in chosen]}, f)

    data = cfg.to_dict()
    data["paths"].update(results_dir=str(results), models_dir=str(cfg.models_dir.parent / f"models_{args.name}"),
                         subset_file=str(subset), embeddings_dir=str(cfg.embeddings_dir))
    data["paths"]["dataset_dirs"] = {k: str(cfg.resolve(v)) for k, v in cfg.paths.dataset_dirs.items()}
    data["paths"]["dataset_dir"] = str(cfg.resolve(cfg.paths.dataset_dir))
    data["labels"] = {"classes": [], "label_aliases": {}, "reject_classes": []}   # the new phase has its own classes
    data["stage"] = {"name": args.name, "title": f"Phase '{args.name}'",
                     "description": f"Only the bees of {sorted(keep)} from '{args.snapshot}'."}
    header = (f"# Stage '{args.name}': {len(chosen)} bees with class {sorted(keep)} in snapshot '{args.snapshot}'\n"
              f"# of {config_path.name}. Written by scripts/new_stage.py; labels and models are kept separate.\n\n")
    out_config.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    human = sum(r["source"] == "human" for r in chosen)
    print(f"Stage '{args.name}': {len(chosen)} bees out of {len(rows)} ({human} labelled by hand, {len(chosen) - human} assigned by the model)")
    print(f"  bee list:         {subset}")
    print(f"  config:           {out_config}")
    print("To use it, start the hub like this (from active_learning_tool/):")
    print(f"  BEE_AL_CONFIG={out_config} ../venv/bin/python -u app.py")


if __name__ == "__main__":
    main()
