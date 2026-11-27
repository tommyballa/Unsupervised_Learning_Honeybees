"""Bring the bees of another folder into a later phase, filtering them with the previous phase's model.

Use case: phase 1 was done on the training set only; to add the test set to phase 2 the same
criterion is applied to it. The new bees get their phase-1 class from its model (same extractor,
same labels, k-NN + propagation as in the hub) and only those of the kept classes pass.

Usage:
    python scripts/add_dataset_to_stage.py --snapshot Prova1 --stage-config config/al_fase2.yaml \
        --name test_b34_novarroa --dir /path/cleaned_bees --keep "Ape Incompleta" "Crop Utilizzabile"

Bees too far from every labelled example are left out, using the same similarity threshold that
separated the "unassigned" training bees in the snapshot. The predictions of all new bees go to
``<results of the previous phase>/added_<name>_predictions.csv``.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.audit import class_normalizer  # noqa: E402
from bee_al.config import DEFAULT_CONFIG_PATH, ID_SEPARATOR, load_config  # noqa: E402
from bee_al.embeddings import (  # noqa: E402
    build_torch_encoder, extract_embeddings, list_dataset, load_backbone, load_embeddings, restrict_to_datasets)
from bee_al.engine import predict_all  # noqa: E402
from bee_al.knn import nearest_labeled_similarity  # noqa: E402
from bee_al.label_store import LabelStore, is_training_event  # noqa: E402
from bee_al.memory_bank import MemoryBank  # noqa: E402
from bee_al.propagation import LabelSpreader  # noqa: E402
from bee_al.retrain import latest_embeddings  # noqa: E402
from bee_al.snapshot import read_csv, read_meta, snapshot_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="snapshot of the previous phase from which the new phase was created")
    ap.add_argument("--stage-config", required=True, help="config of the phase to add the bees to")
    ap.add_argument("--name", required=True, help="name of the dataset folder to add (image_id prefix)")
    ap.add_argument("--dir", required=True, help="folder with the cleaned crops to add")
    ap.add_argument("--keep", required=True, nargs="+", help="classes of the previous phase to keep")
    ap.add_argument("--config", default=None, help="config of the previous phase (default config/al_config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    stage_path = Path(args.stage_config)
    stage = load_config(stage_path)
    if stage.subset_path is None:
        sys.exit(f"{stage_path} has no paths.subset_file: it is not a phase built on a subset")
    folder = snapshot_path(cfg, args.snapshot)
    meta = read_meta(folder)
    keep = {class_normalizer(cfg)(c) for c in args.keep}

    # 1. Embeddings of the previous phase: those of its model, plus the new bees with the same weights.
    version, emb_dir = latest_embeddings(cfg)
    if version != meta["model_version"]:
        sys.exit(f"The snapshot is from model {meta['model_version']} but the previous phase now uses {version}")
    old = restrict_to_datasets(load_embeddings(emb_dir), list(cfg.paths.dataset_dirs))
    new_ids, new_paths = list_dataset([(f"{args.name}{ID_SEPARATOR}", Path(args.dir))])
    if not new_ids:
        sys.exit(f"No images in {args.dir}")
    if set(new_ids) & set(old.ids):
        sys.exit(f"The bees of '{args.name}' are already part of the previous phase")
    out = cfg.results_dir / f"added_{args.name}_embeddings_{version}"
    backbone = load_backbone(cfg.extractor.model)
    weights = cfg.models_dir / version / "backbone.pt"
    if weights.exists():
        import torch

        backbone.load_state_dict(torch.load(weights, map_location="cpu"))
    ex = cfg.extractor
    encoder = build_torch_encoder(backbone, ex.image_size, ex.batch_size, ex.num_workers, ex.device)
    # Check: the same weights must reproduce the saved embeddings of the previous phase.
    probe = np.random.default_rng(0).choice(len(old.ids), size=32, replace=False)
    again = encoder([Path(old.paths[i]) for i in probe])
    again /= np.linalg.norm(again, axis=1, keepdims=True)
    parity = float((again * old.X[probe]).sum(axis=1).min())
    print(f"Model {version}: minimum cosine with the saved embeddings {parity:.4f}")
    if parity < 0.99:
        sys.exit("The weights do not reproduce the embeddings of the previous phase: stopping")
    extract_embeddings(new_paths, out, encoder, meta={"model": cfg.extractor.model, "weights": version},
                       shard_size=ex.shard_size, log=print, image_ids=new_ids)
    new = load_embeddings(out)

    # 2. Same labels and same procedure as the hub, on old and new bees together.
    rows = {r["image_id"]: r for r in read_csv(folder / "predictions.csv")}
    validation = {i for i, r in rows.items() if r["status"] == "validation"}
    store = LabelStore(folder / "al_labels.csv", cfg.labels.label_aliases)   # the labels frozen in the snapshot
    bank = MemoryBank(old.ids + new.ids, np.concatenate([old.X, new.X]), old.paths + new.paths, backend=cfg.knn.backend)
    for image_id, ev in store.current(is_training_event).items():
        if image_id in bank and image_id not in validation:
            bank.add_label(image_id, ev.label)
    p = cfg.propagation
    spreader = LabelSpreader.from_bank(bank, p.n_neighbors, p.alpha, p.max_iter, p.tol) if p.enabled else None
    _, final = predict_all(bank, spreader, cfg)
    labels, confidence = final.labels, final.confidence
    similarity = nearest_labeled_similarity(bank)

    # The "unassigned" ones: same similarity threshold that separated the training ones in the snapshot.
    unassigned = [float(r["nearest_labeled_similarity"]) for r in rows.values() if r["status"] == "unassigned"]
    threshold = max(unassigned) if unassigned else -1.0
    agree = np.mean([labels[bank.index_of(i)] == r["final_label"] for i, r in rows.items() if r["status"] == "predicted"])
    print(f"Already known bees: same class as the snapshot in {100 * agree:.1f}% of cases (with the new bees in the graph)")

    table, chosen = [], []
    for image_id in new.ids:
        i = bank.index_of(image_id)
        status = "unassigned" if similarity[i] <= threshold else "predicted"
        if status == "predicted" and labels[i] in keep:
            chosen.append(image_id)
        table.append({"image_id": image_id, "status": status, "label": labels[i] or "", "confidence": f"{confidence[i]:.6f}",
                      "nearest_labeled_similarity": f"{similarity[i]:.6f}", "kept": int(status == "predicted" and labels[i] in keep)})
    report = cfg.results_dir / f"added_{args.name}_predictions.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"New bees: {len(new.ids)}; per class: {dict(Counter(r['label'] for r in table if r['status'] == 'predicted'))}; "
          f"unassigned {sum(r['status'] == 'unassigned' for r in table)}")

    # 3. They are added to the phase: bee list and dataset folder in its config.
    with open(stage.subset_path, "r", encoding="utf-8") as f:
        subset = json.load(f)
    before = len(subset["image_ids"])
    subset["image_ids"] = list(dict.fromkeys(subset["image_ids"] + chosen))
    subset.setdefault("added", []).append({"dataset": args.name, "dir": str(args.dir), "model_version": version, "keep": sorted(keep),
                                           "n_candidates": len(new.ids), "n_added": len(subset["image_ids"]) - before})
    with open(stage.subset_path, "w", encoding="utf-8") as f:
        json.dump(subset, f)
    text = stage_path.read_text(encoding="utf-8")
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    data = yaml.safe_load(text)
    data["paths"]["dataset_dirs"][args.name] = str(Path(args.dir))
    note = f"# Added {len(subset['image_ids']) - before} bees of '{args.name}' filtered with model {version} of the previous phase.\n"
    stage_path.write_text(header + note + "\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Kept {len(chosen)} bees out of {len(new.ids)}: the phase goes from {before} to {len(subset['image_ids'])} bees")
    print(f"  predictions: {report}\n  list:        {stage.subset_path}\n  config:      {stage_path}")


if __name__ == "__main__":
    main()
