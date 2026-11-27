"""Simula l'hub con un oracolo e produce le curve di apprendimento delle varianti.

Uso:
    python scripts/simulate.py                                   # dataset sintetico
    python scripts/simulate.py --oracle store                    # etichette di al_labels.csv
    python scripts/simulate.py --oracle csv:etichette.csv        # CSV con colonne image_id,label
    python scripts/simulate.py --variants baseline v1 v2 v3 --seeds 0 1 2

L'oracolo può essere un insieme qualsiasi di etichette: le classi non sono fissate.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al import UNKNOWN_LABEL  # noqa: E402
from bee_al.config import load_config  # noqa: E402
from bee_al.embeddings import load_embeddings  # noqa: E402
from bee_al.label_store import LabelStore  # noqa: E402
from bee_al.retrain import latest_embeddings  # noqa: E402
from bee_al.simulation import VARIANTS, make_synthetic, oracle_subset, run_simulation, save_results  # noqa: E402


def load_oracle(cfg, source: str):
    """Embedding e etichette dell'oracolo scelto."""
    sim = cfg.simulation
    if source == "synthetic":
        return make_synthetic(sim.synthetic_n, sim.synthetic_dim, sim.synthetic_classes,
                              sim.synthetic_imbalance, sim.synthetic_noise, seed=cfg.seed)
    store = LabelStore(cfg.al_labels_path, cfg.labels.label_aliases)
    if source == "store":
        labels = {i: ev.label for i, ev in store.current().items()}
    elif source.startswith("csv:"):
        with open(source[4:], "r", encoding="utf-8") as f:
            labels = {r["image_id"]: store.resolve(store.normalize_label(r["label"])) for r in csv.DictReader(f)}
    else:
        sys.exit(f"Oracolo sconosciuto: {source}")
    labels = {i: lbl for i, lbl in labels.items() if lbl != UNKNOWN_LABEL}
    emb = load_embeddings(latest_embeddings(cfg)[1])
    return oracle_subset(emb.ids, emb.X, labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--oracle", default="synthetic", help="synthetic | store | csv:<file>")
    ap.add_argument("--variants", nargs="+", default=["baseline", "v1", "v2", "v3"], choices=sorted(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-labels", type=int, default=None)
    ap.add_argument("--out", default="runs/simulation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_labels:
        cfg.simulation.max_labels = args.max_labels
    X, oracle = load_oracle(cfg, args.oracle)
    if len(X) < 2 * cfg.query.batch_size:
        sys.exit(f"L'oracolo copre solo {len(X)} api: troppo poche per simulare")
    print(f"Oracolo '{args.oracle}': {len(X)} api, {len(set(oracle))} classi")

    result = run_simulation(X, oracle, cfg, args.variants, args.seeds, log=print)
    paths = save_results(result, cfg.resolve(args.out))
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
