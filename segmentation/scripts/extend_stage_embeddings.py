"""Aggiunge agli embedding del modello corrente di una fase le api di una cartella appena entrata nella fase.

Caso d'uso: un batch nuovo (es. tenuto da parte per la validazione) entra in una fase il cui extractor
è già stato riaddestrato. Gli embedding delle api nuove si calcolano con gli stessi pesi e si aggiungono
come shard in più alla cartella della versione corrente, così l'hub li carica insieme agli altri.

Uso:
    python scripts/extend_stage_embeddings.py --config config/al_fase2_sam3.yaml --name b7

Si aggiungono solo le api di ``--name`` presenti nell'elenco della fase (``paths.subset_file``), o tutte se
la fase non ha un elenco. Prima di scrivere si controlla che i pesi riproducano gli embedding salvati;
il manifest originale resta in ``manifest.before_<name>.json``. Si può interrompere e rilanciare.
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
    ap.add_argument("--config", required=True, help="config della fase")
    ap.add_argument("--name", required=True, help="cartella di dataset (chiave di paths.dataset_dirs) da aggiungere")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.name not in cfg.paths.dataset_dirs:
        sys.exit(f"'{args.name}' non è in paths.dataset_dirs di {args.config}")
    version, emb_dir = latest_embeddings(cfg)
    manifest = read_manifest(emb_dir)
    if manifest is None:
        sys.exit(f"Nessun embedding in {emb_dir}")

    ids, paths = list_dataset([(f"{args.name}{ID_SEPARATOR}", cfg.resolve(cfg.paths.dataset_dirs[args.name]))])
    if cfg.subset_path is not None:
        with open(cfg.subset_path, "r", encoding="utf-8") as f:
            wanted = set(json.load(f)["image_ids"])
        ids, paths = map(list, zip(*[(i, p) for i, p in zip(ids, paths) if i in wanted])) if wanted & set(ids) else ([], [])
    if not ids:
        sys.exit(f"Nessuna api di '{args.name}' da aggiungere")
    print(f"Versione {version} ({emb_dir}): {len(ids)} api di '{args.name}' nella fase")

    old = load_embeddings(emb_dir)
    old_ids = set(old.ids)
    backbone = load_backbone(cfg.extractor.model)
    if version != BASE_VERSION:
        import torch

        backbone.load_state_dict(torch.load(cfg.models_dir / version / "backbone.pt", map_location="cpu"))
    ex = cfg.extractor
    encoder = build_torch_encoder(backbone, ex.image_size, ex.batch_size, ex.num_workers, ex.device)
    # Controllo: gli stessi pesi devono riprodurre gli embedding già salvati.
    known = [j for j, i in enumerate(old.ids) if not i.startswith(f"{args.name}{ID_SEPARATOR}")]
    probe = np.random.default_rng(0).choice(known, size=min(32, len(known)), replace=False)
    again = encoder([Path(old.paths[j]) for j in probe])
    again /= np.linalg.norm(again, axis=1, keepdims=True)
    parity = float((again * old.X[probe]).sum(axis=1).min())
    print(f"Coseno minimo con gli embedding salvati: {parity:.4f}")
    if parity < 0.99:
        sys.exit("I pesi non riproducono gli embedding salvati: mi fermo")

    backup = emb_dir / f"manifest.before_{args.name}.json"
    if not backup.exists():
        shutil.copy2(emb_dir / MANIFEST_NAME, backup)
    added = extract_embeddings(paths, emb_dir, encoder, meta=manifest["meta"], shard_size=ex.shard_size,
                               log=print, image_ids=ids)
    total = len(load_embeddings(emb_dir).ids)
    print(f"Aggiunti {added} embedding ({len(old_ids & set(ids))} c'erano già); ora {total} in {emb_dir}")


if __name__ == "__main__":
    main()
