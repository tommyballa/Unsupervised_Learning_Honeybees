"""Add the bees of a folder that has just entered a phase to the embeddings of that phase's current model.

Use case: a new batch (e.g. held out for validation) enters a phase whose extractor has already been
re-trained. The embeddings of the new bees are computed with the same weights and added as extra
shards to the current version's folder, so the hub loads them together with the others.

Usage:
    python scripts/extend_stage_embeddings.py --config config/al_fase2_sam3.yaml --name b7

Only the bees of ``--name`` that are in the phase's list (``paths.subset_file``) are added, or all of them if
the phase has no list. Before writing, the weights are checked to reproduce the saved embeddings;
the original manifest is kept in ``manifest.before_<name>.json``. It can be interrupted and restarted.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import ID_SEPARATOR, load_config  # noqa: E402
from bee_al.embeddings import (  # noqa: E402
    MANIFEST_NAME, build_torch_encoder, extract_embeddings, list_dataset, load_backbone, load_embeddings, read_manifest)
from bee_al.retrain import BASE_VERSION, latest_embeddings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config of the phase")
    ap.add_argument("--name", required=True, help="dataset folder (key of paths.dataset_dirs) to add")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name not in cfg.paths.dataset_dirs:
        sys.exit(f"'{args.name}' is not in paths.dataset_dirs of {args.config}")
    version, emb_dir = latest_embeddings(cfg)
    manifest = read_manifest(emb_dir)
    if manifest is None:
        sys.exit(f"No embeddings in {emb_dir}")

    ids, paths = list_dataset([(f"{args.name}{ID_SEPARATOR}", cfg.resolve(cfg.paths.dataset_dirs[args.name]))])
    if cfg.subset_path is not None:
        with open(cfg.subset_path, "r", encoding="utf-8") as f:
            wanted = set(json.load(f)["image_ids"])
        ids, paths = map(list, zip(*[(i, p) for i, p in zip(ids, paths) if i in wanted])) if wanted & set(ids) else ([], [])
    if not ids:
        sys.exit(f"No bees of '{args.name}' to add")
    print(f"Version {version} ({emb_dir}): {len(ids)} bees of '{args.name}' in the phase")

    old = load_embeddings(emb_dir)
    old_ids = set(old.ids)
    backbone = load_backbone(cfg.extractor.model)
    if version != BASE_VERSION:
        import torch

        backbone.load_state_dict(torch.load(cfg.models_dir / version / "backbone.pt", map_location="cpu"))
    ex = cfg.extractor
    encoder = build_torch_encoder(backbone, ex.image_size, ex.batch_size, ex.num_workers, ex.device)
    # Check: the same weights must reproduce the already saved embeddings.
    known = [j for j, i in enumerate(old.ids) if not i.startswith(f"{args.name}{ID_SEPARATOR}")]
    probe = np.random.default_rng(0).choice(known, size=min(32, len(known)), replace=False)
    again = encoder([Path(old.paths[j]) for j in probe])
    again /= np.linalg.norm(again, axis=1, keepdims=True)
    parity = float((again * old.X[probe]).sum(axis=1).min())
    print(f"Minimum cosine with the saved embeddings: {parity:.4f}")
    if parity < 0.99:
        sys.exit("The weights do not reproduce the saved embeddings: stopping")

    backup = emb_dir / f"manifest.before_{args.name}.json"
    if not backup.exists():
        shutil.copy2(emb_dir / MANIFEST_NAME, backup)
    added = extract_embeddings(paths, emb_dir, encoder, meta=manifest["meta"], shard_size=ex.shard_size,
                               log=print, image_ids=ids)
    total = len(load_embeddings(emb_dir).ids)
    print(f"Added {added} embeddings ({len(old_ids & set(ids))} were already there); now {total} in {emb_dir}")


if __name__ == "__main__":
    main()
