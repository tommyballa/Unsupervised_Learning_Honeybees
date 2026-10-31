"""Porta in una fase successiva le api di un'altra cartella, filtrandole con il modello della fase precedente.

Caso d'uso: la fase 1 è stata fatta sul solo training set; per aggiungere il test set alla fase 2 gli si
applica lo stesso criterio. Le api nuove ricevono la classe della fase 1 dal suo modello (stesso extractor,
stesse etichette, k-NN + propagazione come nell'hub) e passano solo quelle delle classi tenute.

Uso:
    python scripts/add_dataset_to_stage.py --snapshot Prova1 --stage-config config/al_fase2.yaml \
        --name test_b34_novarroa --dir /percorso/cleaned_bees --keep "Ape Incompleta" "Crop Utilizzabile"

Le api troppo lontane da ogni esempio etichettato restano fuori, con la stessa soglia di similarità che
nello snapshot separava le "non assegnate" del training. Le predizioni di tutte le api nuove finiscono in
``<results della fase precedente>/added_<name>_predictions.csv``.
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
    ap.add_argument("--snapshot", required=True, help="snapshot della fase precedente da cui è nata la fase nuova")
    ap.add_argument("--stage-config", required=True, help="config della fase a cui aggiungere le api")
    ap.add_argument("--name", required=True, help="nome della cartella di dataset da aggiungere (prefisso degli image_id)")
    ap.add_argument("--dir", required=True, help="cartella con i crop puliti da aggiungere")
    ap.add_argument("--keep", required=True, nargs="+", help="classi della fase precedente da tenere")
    ap.add_argument("--config", default=None, help="config della fase precedente (default config/al_config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    stage_path = Path(args.stage_config)
    stage = load_config(stage_path)
    if stage.subset_path is None:
        sys.exit(f"{stage_path} non ha un paths.subset_file: non è una fase costruita su un sottoinsieme")
    folder = snapshot_path(cfg, args.snapshot)
    meta = read_meta(folder)
    keep = {class_normalizer(cfg)(c) for c in args.keep}

    # 1. Embedding della fase precedente: quelli del suo modello, più le api nuove con gli stessi pesi.
    version, emb_dir = latest_embeddings(cfg)
    if version != meta["model_version"]:
        sys.exit(f"Lo snapshot è del modello {meta['model_version']} ma la fase precedente ora usa {version}")
    old = restrict_to_datasets(load_embeddings(emb_dir), list(cfg.paths.dataset_dirs))
    new_ids, new_paths = list_dataset([(f"{args.name}{ID_SEPARATOR}", Path(args.dir))])
    if not new_ids:
        sys.exit(f"Nessuna immagine in {args.dir}")
    if set(new_ids) & set(old.ids):
        sys.exit(f"Le api di '{args.name}' fanno già parte della fase precedente")
    out = cfg.results_dir / f"added_{args.name}_embeddings_{version}"
    backbone = load_backbone(cfg.extractor.model)
    weights = cfg.models_dir / version / "backbone.pt"
    if weights.exists():
        import torch

        backbone.load_state_dict(torch.load(weights, map_location="cpu"))
    ex = cfg.extractor
    encoder = build_torch_encoder(backbone, ex.image_size, ex.batch_size, ex.num_workers, ex.device)
    # Controllo: gli stessi pesi devono riprodurre gli embedding salvati della fase precedente.
    probe = np.random.default_rng(0).choice(len(old.ids), size=32, replace=False)
    again = encoder([Path(old.paths[i]) for i in probe])
    again /= np.linalg.norm(again, axis=1, keepdims=True)
    parity = float((again * old.X[probe]).sum(axis=1).min())
    print(f"Modello {version}: coseno minimo con gli embedding salvati {parity:.4f}")
    if parity < 0.99:
        sys.exit("I pesi non riproducono gli embedding della fase precedente: mi fermo")
    extract_embeddings(new_paths, out, encoder, meta={"model": cfg.extractor.model, "weights": version},
                       shard_size=ex.shard_size, log=print, image_ids=new_ids)
    new = load_embeddings(out)

    # 2. Stesse etichette e stessa procedura dell'hub, su api vecchie e nuove insieme.
    rows = {r["image_id"]: r for r in read_csv(folder / "predictions.csv")}
    validation = {i for i, r in rows.items() if r["status"] == "validation"}
    store = LabelStore(folder / "al_labels.csv", cfg.labels.label_aliases)   # le etichette congelate nello snapshot
    bank = MemoryBank(old.ids + new.ids, np.concatenate([old.X, new.X]), old.paths + new.paths, backend=cfg.knn.backend)
    for image_id, ev in store.current(is_training_event).items():
        if image_id in bank and image_id not in validation:
            bank.add_label(image_id, ev.label)
    p = cfg.propagation
    spreader = LabelSpreader.from_bank(bank, p.n_neighbors, p.alpha, p.max_iter, p.tol) if p.enabled else None
    _, final = predict_all(bank, spreader, cfg)
    labels, confidence = final.labels, final.confidence
    similarity = nearest_labeled_similarity(bank)

    # Le "non assegnate": stessa soglia di similarità che nello snapshot separava quelle del training.
    unassigned = [float(r["nearest_labeled_similarity"]) for r in rows.values() if r["status"] == "unassigned"]
    threshold = max(unassigned) if unassigned else -1.0
    agree = np.mean([labels[bank.index_of(i)] == r["final_label"] for i, r in rows.items() if r["status"] == "predicted"])
    print(f"Api già note: stessa classe dello snapshot nel {100 * agree:.1f}% dei casi (con le api nuove nel grafo)")

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
    print(f"Api nuove: {len(new.ids)}; per classe: {dict(Counter(r['label'] for r in table if r['status'] == 'predicted'))}; "
          f"non assegnate {sum(r['status'] == 'unassigned' for r in table)}")

    # 3. Si aggiungono alla fase: elenco delle api e cartella di dataset nella sua config.
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
    note = f"# Aggiunte {len(subset['image_ids']) - before} api di '{args.name}' filtrate con il modello {version} della fase precedente.\n"
    stage_path.write_text(header + note + "\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Tenute {len(chosen)} api su {len(new.ids)}: la fase passa da {before} a {len(subset['image_ids'])} api")
    print(f"  predizioni: {report}\n  elenco:     {stage.subset_path}\n  config:     {stage_path}")


if __name__ == "__main__":
    main()
