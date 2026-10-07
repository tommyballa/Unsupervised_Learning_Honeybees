"""Prepara la fase successiva della pipeline partendo da uno snapshot.

La fase 1 separa, per esempio, le api segmentate bene da quelle segmentate male. La fase 2
lavora solo sulle api che la fase 1 ha tenuto: questo script le elenca e scrive una config
nuova, con etichette e modelli in cartelle proprie, così le due fasi non si mescolano.

Uso:
    python scripts/new_stage.py --snapshot fase1_finale --keep "Segmentata Bene" --name fase2
    BEE_AL_CONFIG=config/al_fase2.yaml ../venv/bin/python -u app.py      # da active_learning_tool/

Entrano nella fase nuova le api dello snapshot con classe finale in --keep, etichettate a mano
o assegnate dal modello (con --min-confidence si scartano le assegnazioni poco sicure).
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
    ap.add_argument("--snapshot", required=True, help="snapshot della fase corrente da cui partire")
    ap.add_argument("--keep", required=True, nargs="+", help="classi da portare nella fase nuova")
    ap.add_argument("--name", required=True, help="nome della fase nuova (lettere, cifre, '_', '-')")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="confidenza minima per le api assegnate dal modello")
    ap.add_argument("--config", default=None, help="config della fase corrente (default config/al_config.yaml)")
    ap.add_argument("--out-config", default=None, help="config da scrivere (default config/al_<name>.yaml)")
    args = ap.parse_args()

    if not args.name.replace("_", "").replace("-", "").isalnum():
        sys.exit("Nome della fase non valido: usa solo lettere, cifre, '_' e '-'")
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    try:
        folder = snapshot_path(cfg, args.snapshot)
    except ValueError as e:
        sys.exit(str(e))
    if not (folder / "predictions.csv").exists():
        sys.exit(f"Snapshot '{args.snapshot}' non trovato in {folder.parent}")

    normalize = class_normalizer(cfg)
    keep = {normalize(c) for c in args.keep}
    rows = read_csv(folder / "predictions.csv")
    present = Counter(r["final_label"] for r in rows if r["final_label"])
    missing = keep - set(present)
    if missing:
        sys.exit(f"Classi non presenti nello snapshot: {sorted(missing)}. Ci sono: {dict(present)}")
    chosen = [r for r in rows if r["final_label"] in keep and r["status"] in ("labeled", "predicted")
              and (r["source"] == "human" or float(r["confidence"]) >= args.min_confidence)]
    if not chosen:
        sys.exit("Nessuna ape supera i criteri: niente da portare nella fase nuova")

    out_config = Path(args.out_config) if args.out_config else REPO_ROOT / "config" / f"al_{args.name}.yaml"
    results = cfg.results_dir.parent / f"results_{args.name}"
    subset = results / "subset.json"
    for path in (out_config, subset):
        if path.exists():
            sys.exit(f"{path} esiste già: scegli un altro nome per la fase")
    results.mkdir(parents=True, exist_ok=True)
    with open(subset, "w", encoding="utf-8") as f:
        json.dump({"from_snapshot": args.snapshot, "from_config": str(config_path), "keep": sorted(keep),
                   "min_confidence": args.min_confidence, "image_ids": [r["image_id"] for r in chosen]}, f)

    data = cfg.to_dict()
    data["paths"].update(results_dir=str(results), models_dir=str(cfg.models_dir.parent / f"models_{args.name}"),
                         subset_file=str(subset), embeddings_dir=str(cfg.embeddings_dir))
    data["paths"]["dataset_dirs"] = {k: str(cfg.resolve(v)) for k, v in cfg.paths.dataset_dirs.items()}
    data["paths"]["dataset_dir"] = str(cfg.resolve(cfg.paths.dataset_dir))
    data["labels"] = {"classes": [], "label_aliases": {}, "reject_classes": []}   # la fase nuova ha classi sue
    header = (f"# Fase '{args.name}': {len(chosen)} api con classe {sorted(keep)} nello snapshot '{args.snapshot}'\n"
              f"# di {config_path.name}. Scritta da scripts/new_stage.py; etichette e modelli sono separati.\n\n")
    out_config.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    human = sum(r["source"] == "human" for r in chosen)
    print(f"Fase '{args.name}': {len(chosen)} api su {len(rows)} ({human} etichettate a mano, {len(chosen) - human} assegnate dal modello)")
    print(f"  elenco delle api: {subset}")
    print(f"  config:           {out_config}")
    print("Per usarla, avvia l'hub così (da active_learning_tool/):")
    print(f"  BEE_AL_CONFIG={out_config} ../venv/bin/python -u app.py")


if __name__ == "__main__":
    main()
