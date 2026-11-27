"""Conteggi del modello per batch (sessione) da uno snapshot, senza correzione dell'audit.

Uso:
    python scripts/batch_counts.py --snapshot fase2_finale --config config/al_fase2_sam3.yaml

Ogni ape conta nella sua classe finale: l'etichetta data a mano se c'è, altrimenti quella
predetta dal modello. Le api del validation set contano con la classe predetta. Restano in
colonne a parte, fuori dalle quote, le classi di scarto (labels.reject_classes), le non
assegnate (troppo lontane da ogni esempio), le "Scarta ape" e quelle senza predizione.

Per ogni batch: foto, api, conteggio di ogni classe, api per foto e due quote:
- "%": sulle api classificate (tutte le classi non di scarto);
- "% profile": solo per colony.demographic_classes, sul totale di quelle classi (profilo demografico).
L'intervallo al 95% delle quote viene da un bootstrap sulle foto del batch: misura quanto la
quota cambia da foto a foto, NON gli errori del modello (per quelli serve l'audit, vedi
scripts/audit_report.py).

Scrive in <snapshot>/batch_counts/: batch_counts.csv (una riga per batch), photo_counts.csv
(una riga per foto) e batch_counts.png.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bee_al.config import load_config  # noqa: E402
from bee_al.snapshot import read_csv, read_meta, snapshot_path  # noqa: E402

OUT_DIR = "batch_counts"
UNASSIGNED, SET_ASIDE, NO_PREDICTION = "non assegnate", "Scarta ape", "senza predizione"
EXTRA = (UNASSIGNED, SET_ASIDE, NO_PREDICTION)


def category(row: dict[str, str]) -> str:
    """Classe con cui l'ape entra nei conteggi."""
    status = row["status"]
    if status in ("labeled", "predicted", "rejected"):
        return row["final_label"]
    if status == "validation":
        return row["propagation_label"] or NO_PREDICTION
    if status == "unassigned":
        return UNASSIGNED
    if status == "unknown":
        return SET_ASIDE
    return NO_PREDICTION


def batch_order(sessions: set[str], meta: dict) -> list[str]:
    """Ordine dei batch: colony.time_order della config dello snapshot, poi gli altri in ordine alfabetico."""
    order: list[str] = []
    for seq in meta.get("config", {}).get("colony", {}).get("time_order", {}).values():
        order += [s for s in seq if s in sessions and s not in order]
    return order + sorted(sessions - set(order))


def _shares_ci(boot: np.ndarray, cols: list[int], n_boot: int) -> tuple[np.ndarray, np.ndarray]:
    """Percentili 2,5 e 97,5 delle quote delle colonne ``cols`` sul loro totale, replica per replica."""
    part = boot[:, cols]
    tot = part.sum(axis=1, keepdims=True)
    shares = np.divide(part, tot, out=np.zeros_like(part), where=tot > 0)
    lo, hi = np.percentile(shares, [2.5, 97.5], axis=0)
    return 100 * lo, 100 * hi


def count(rows: list[dict[str, str]], frames: list[dict[str, str]], meta: dict, n_boot: int = 1000, seed: int = 0,
          reject: tuple[str, ...] = (), demographic: tuple[str, ...] = ()):
    """(righe per batch, righe per foto, classi). Le foto senza api accettate contano comunque come foto."""
    photos: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for f in frames:
        photos[(f["session"], f["frame"])]
    for r in rows:
        photos[(r["session"], r["frame"])][category(r)] += 1
    seen = {c for counts in photos.values() for c in counts}
    classes = sorted(seen - set(EXTRA) - set(reject))
    reject_cols = [c for c in reject if c in seen]
    columns = classes + reject_cols + list(EXTRA)
    profile = [j for j, c in enumerate(classes) if c in demographic]

    rng = np.random.default_rng(seed)
    by_batch: list[dict] = []
    for batch in batch_order({s for s, _ in photos}, meta):
        keys = sorted(k for k in photos if k[0] == batch)
        C = np.array([[photos[k][c] for c in columns] for k in keys], dtype=float)
        total, n_photos = C.sum(axis=0), len(keys)
        in_classes = total[:len(classes)].sum()
        out = {"batch": batch, "photos": n_photos, "bees": int(total.sum()), "bees_in_classes": int(in_classes)}
        boot = C[rng.integers(0, n_photos, (n_boot, n_photos))].sum(axis=1)
        lo, hi = _shares_ci(boot, list(range(len(classes))), n_boot)
        for j, c in enumerate(classes):
            out.update({f"{c}": int(total[j]), f"{c} %": 100 * total[j] / in_classes if in_classes else 0.0,
                        f"{c} % lo": lo[j], f"{c} % hi": hi[j], f"{c} per photo": total[j] / n_photos})
        if profile:
            plo, phi = _shares_ci(boot, profile, n_boot)
            in_profile = total[profile].sum()
            for i, j in enumerate(profile):
                c = classes[j]
                out.update({f"{c} % profile": 100 * total[j] / in_profile if in_profile else 0.0,
                            f"{c} % profile lo": plo[i], f"{c} % profile hi": phi[i]})
        for j, c in enumerate(reject_cols + list(EXTRA), start=len(classes)):
            out[c] = int(total[j])
        out[f"{UNASSIGNED} % of bees"] = 100 * out[UNASSIGNED] / out["bees"] if out["bees"] else 0.0
        by_batch.append(out)
    per_photo = [{"batch": s, "frame": f, "bees": sum(photos[(s, f)].values()), **{c: photos[(s, f)][c] for c in columns}}
                 for s, f in sorted(photos)]
    return by_batch, per_photo, classes


def plot(by_batch: list[dict], classes: list[str], path: Path, demographic: tuple[str, ...] = ()) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(by_batch))
    labels = [b["batch"] for b in by_batch]
    profile = [c for c in classes if c in demographic]
    others = [c for c in classes if c not in demographic] if profile else classes
    panels = ([("% profile", profile, "Profilo demografico", "% di " + " + ".join(profile))] if profile else []) + \
             ([("%", others, "Altre classi" if profile else "Quota per classe", "% delle api classificate")] if others else [])
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6.5 * (len(panels) + 1), 5))
    for ax, (key, cols, title, ylabel) in zip(axes, panels):
        for c in cols:
            y = np.array([b[f"{c} {key}"] for b in by_batch])
            err = np.array([[y[i] - b[f"{c} {key} lo"], b[f"{c} {key} hi"] - y[i]] for i, b in enumerate(by_batch)]).T
            ax.errorbar(x, y, yerr=err, marker="o", capsize=4, label=c)
        ax.set(title=f"{title} (bootstrap sulle foto, 95%)", ylabel=ylabel)
    for c in classes:
        axes[-1].plot(x, [b[f"{c} per photo"] for b in by_batch], marker="o", label=c)
    axes[-1].set(title="Api per foto", ylabel="api / foto")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Conteggi del modello per batch (senza correzione dell'audit)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="nome dello snapshot")
    ap.add_argument("--config", default=None, help="config della fase (default config/al_config.yaml)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    folder = snapshot_path(cfg, args.snapshot)
    if not (folder / "predictions.csv").exists():
        sys.exit(f"Snapshot '{args.snapshot}' non trovato in {folder.parent}")
    meta = read_meta(folder)
    reject = tuple(dict.fromkeys(list(meta.get("reject_classes", [])) + cfg.labels.reject_classes))
    demographic = tuple(cfg.colony.demographic_classes)
    by_batch, per_photo, classes = count(read_csv(folder / "predictions.csv"), read_csv(folder / "frames.csv"),
                                         meta, args.n_boot, reject=reject, demographic=demographic)
    out = folder / OUT_DIR
    out.mkdir(exist_ok=True)
    write_csv(out / "batch_counts.csv", by_batch)
    write_csv(out / "photo_counts.csv", per_photo)
    plot(by_batch, classes, out / "batch_counts.png", demographic)

    print(f"Conteggi del modello per batch, snapshot '{args.snapshot}' (senza audit)\n")
    extras = [k for k in reject if k in by_batch[0]] + [UNASSIGNED]
    print("batch     foto    api  " + "  ".join(f"{c:>22}" for c in classes) + "  " + "  ".join(f"{e:>18}" for e in extras))
    for b in by_batch:
        cells = [f"{b[c]:>6} {b[f'{c} %']:5.1f}% [{b[f'{c} % lo']:4.1f}-{b[f'{c} % hi']:4.1f}]" for c in classes]
        print(f"{b['batch']:<8} {b['photos']:>5} {b['bees']:>6}  " + "  ".join(cells) + "  " + "  ".join(f"{b[e]:>18}" for e in extras))
    if demographic:
        print("\nProfilo demografico (% sul totale di " + ", ".join(demographic) + "):")
        for b in by_batch:
            print(f"{b['batch']:<8} " + "  ".join(
                f"{c} {b[f'{c} % profile']:5.1f}% [{b[f'{c} % profile lo']:4.1f}-{b[f'{c} % profile hi']:4.1f}]"
                for c in classes if f"{c} % profile" in b))
    print(f"\nFile in {out}")


if __name__ == "__main__":
    main()
