"""Thesis figures and metrics for the 22 phone photos: segmentation, domain distance, dedicated phase 2.

Reads the results already produced (logs/phone/*.json|csv, snapshot 'foto_telefono', UMAP maps) and writes to OUT:
- metrics.json: all the numbers cited in docs/phone_photos.md;
- fig1_seg_per_foto.png, fig2_similarita_dominio.png, fig3_fase1_vs_fase_telefono.png,
  fig4_composizione_per_foto.png, map_overview.png, map_by_cluster.png, crops_<classe>.png.
Usage: python logs/phone/thesis_report_phone.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bee_al.embeddings import load_embeddings  # noqa: E402
from embedding_map import BACKGROUND, GREY, INK, MUTED, SERIES, english, plot  # noqa: E402
from phone_map import crop_grid  # noqa: E402  (logs/phone/phone_map.py)

AL = Path("/mnt/disk1/tommaso_ballarin/bee_al")
SNAP = AL / "results_fase2_phone/snapshots/foto_telefono"
OUT = SNAP / "report_tesi"
PHONE_DIR = Path("/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee_phone_SAM3/cleaned_bees")
# Same colours as the phase 2 maps (fixed name order with Fuco and Regina): the colour follows the class.
CLASS_ORDER = ["Adulte", "Fuco", "Giovane", "Malata", "Non Classificabile"]
COLOUR = {c: SERIES[k] for k, c in enumerate(CLASS_ORDER)}
UNASSIGNED = "non assegnate"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(plt):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.edgecolor": BACKGROUND, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def loo_knn(X, labels, k=10, temperature=0.07):
    """Weighted leave-one-out k-NN (like bee_al.knn) on the hand-labelled bees only."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    classes = sorted(set(labels))
    y = np.array([classes.index(c) for c in labels])
    pred = []
    for i in range(len(X)):
        nn = np.argsort(-S[i])[:k]
        w = np.exp(S[i, nn] / temperature)
        pred.append(int(np.argmax(np.bincount(y[nn], weights=w, minlength=len(classes)))))
    pred = np.array(pred)
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rec = cm.diagonal() / cm.sum(1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0)
    return {"classes": classes, "accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "per_class": {c: {"n": int(cm[i].sum()), "precision": round(float(prec[i]), 3), "recall": round(float(rec[i]), 3),
                              "f1": round(float(f1[i]), 3)} for i, c in enumerate(classes)},
            "confusion_rows_true_cols_pred": cm.tolist()}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    OUT.mkdir(exist_ok=True)
    m = {}

    # 1. Segmentation
    m["seg_eval_labels_ruotate"] = json.load(open(ROOT / "logs/phone/phone_eval_ruotate.json"))
    m["seg_eval_labels_originali"] = json.load(open(ROOT / "logs/phone/phone_eval.json"))
    m["seg_test_batch3_6"] = json.load(open(ROOT / "logs/ablation_sam3/abl_img1376.json"))
    per = read(ROOT / "logs/phone/phone_compare_ruotate.csv")
    tot = {k: sum(int(r[k]) for r in per) for k in ("sam3", "yolo_c35", "tp_c35", "yolo_c10", "tp_c10")}
    m["seg_match_iou50"] = {**tot, "precision_c35": tot["tp_c35"] / tot["yolo_c35"], "recall_c35": tot["tp_c35"] / tot["sam3"],
                            "precision_c10": tot["tp_c10"] / tot["yolo_c10"], "recall_c10": tot["tp_c10"] / tot["sam3"],
                            "iou_medio_c35_mediana_foto": float(np.median([float(r["iou_medio_c35"]) for r in per]))}
    per.sort(key=lambda r: float(r["rec_c35"]))
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    y = np.arange(len(per))
    ax.hlines(y, [min(float(r["rec_c35"]), float(r["prec_c35"])) for r in per],
              [max(float(r["rec_c35"]), float(r["prec_c35"])) for r in per], color=BACKGROUND, lw=2, zorder=1)
    ax.scatter([float(r["prec_c35"]) for r in per], y, s=40, c=SERIES[0], label="Precision", zorder=2,
               edgecolors="white", linewidths=1.5)
    ax.scatter([float(r["rec_c35"]) for r in per], y, s=40, c=SERIES[1], label="Recall", zorder=2,
               edgecolors="white", linewidths=1.5)
    ax.set_yticks(y, [f"{r['foto'].replace('IMG_20261011_', '')}  ({r['sam3']} bees)" for r in per], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Mask matching with SAM 3 at IoU ≥ 0.5 (YOLO conf ≥ 0.35)")
    ax.grid(axis="x", color=BACKGROUND, lw=0.6)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("YOLO on the 22 phone photos, per photo", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_seg_per_foto.png")
    plt.close(fig)

    # 2. Domain distance: similarity to the nearest labelled example of phase 1 (batches 3-6)
    f1_pred = read(AL / "results_sam3_b3456/snapshots/fase1_finale_okvsscarta/predictions.csv")
    sim_known = np.array([float(r["nearest_labeled_similarity"]) for r in f1_pred if r["nearest_labeled_similarity"]])
    thr = max(float(r["nearest_labeled_similarity"]) for r in f1_pred if r["status"] == "unassigned")
    groups = {"batch 3-6 (phase-1 data)": sim_known}
    added = {}
    for name in ("b7", "phone"):
        added[name] = read(AL / f"results_sam3_b3456/added_{name}_predictions.csv")
        groups[{"b7": "batch 7", "phone": "phone photos"}[name]] = np.array([float(r["nearest_labeled_similarity"]) for r in added[name]])
    m["fase1_soglia_non_assegnate"] = thr
    m["fase1_filtro"] = {}
    for name, rows in added.items():
        c = Counter("non assegnate" if r["status"] == "unassigned" else r["label"] for r in rows)
        m["fase1_filtro"][name] = {"api": len(rows), **c, "quota_non_assegnate": c["non assegnate"] / len(rows),
                                   "similarita_mediana": float(np.median([float(r["nearest_labeled_similarity"]) for r in rows]))}
    m["fase1_filtro"]["batch3_6_similarita_mediana"] = float(np.median(sim_known))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bins = np.linspace(0.3, 1.0, 71)
    # Batch 7 stays in the metrics but not in the figure; the colours stay those of the first version.
    for name, k in (("batch 3-6 (phase-1 data)", 0), ("phone photos", 2)):
        v = groups[name]
        h, _ = np.histogram(v, bins=bins)
        ax.step(bins[:-1], h / h.sum(), where="post", color=SERIES[k], lw=2, label=f"{name} (median {np.median(v):.2f})")
    ax.axvline(thr, color=MUTED, lw=1, ls="--")
    ax.text(thr - 0.005, ax.get_ylim()[1] * 0.55, "unassigned\nthreshold ", color=MUTED, ha="right", va="top", fontsize=9)
    ax.set_xlabel("Cosine similarity to the nearest hand-labelled bee of phase 1 (DINOv2)")
    ax.set_ylabel("Fraction of bees")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Phone bees are far from everything labelled on the camera photos", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_similarita_dominio.png")
    plt.close(fig)

    # 3. Dedicated phase: final classes, labels, re-training, stability
    pred = read(SNAP / "predictions.csv")
    labels = read(SNAP / "al_labels.csv")
    final = {}
    for r in pred:
        if r["status"] == "unassigned":
            final[r["image_id"]] = UNASSIGNED
        elif r["final_label"]:
            final[r["image_id"]] = r["final_label"]
        else:   # validation set: predicted class, as in scripts/batch_counts.py
            final[r["image_id"]] = r["propagation_label"] or r["knn_label"] or UNASSIGNED
    m["fase_telefono"] = {
        "snapshot": json.load(open(SNAP / "meta.json"))["created"],
        "status": dict(Counter(r["status"] for r in pred)),
        "classi_finali": dict(Counter(final.values())),
        "a_mano": dict(Counter(r["label"] for r in labels)),
        "a_mano_per_query": dict(Counter(r["query_type"] for r in labels)),
        "predette": dict(Counter(r["final_label"] for r in pred if r["source"] == "predicted")),
        "rounds": [json.loads(line) for line in open(SNAP.parent.parent / "al_rounds.jsonl")],
        "retrain": {v: {k: json.load(open(AL / f"models_fase2_phone/{v}/train_log.json"))[k]
                        for k in ("n_labeled", "n_pseudo", "loss_first", "loss_last", "duration_s")} for v in ("v001", "v002")},
        "review": dict(Counter(f"{d['decision']}: {d['old_label']} (suggerita {d['suggested']})"
                               for d in map(json.loads, open(SNAP.parent.parent / "review_log.jsonl")))),
        "conf_mediana_predette": {c: float(np.median([float(r["confidence"]) for r in pred if r["source"] == "predicted" and r["final_label"] == c]))
                                  for c in set(r["final_label"] for r in pred if r["source"] == "predicted")},
    }
    for r in m["fase_telefono"]["rounds"]:
        r.pop("classes", None), r.pop("new_classes", None)
    m["batch_counts"] = read(SNAP / "batch_counts/batch_counts.csv")
    m["batch_counts_fase2_batch3_6"] = read(AL / "results_fase2_sam3/snapshots/fase2_finale/batch_counts/batch_counts.csv")

    # 4. Internal consistency: leave-one-out k-NN on the hand labels (original embeddings and v002)
    lab = {r["image_id"]: r["label"] for r in labels}
    m["loo_knn"] = {}
    for v, d in (("v000", AL / "embeddings_phone"), ("v002", AL / "models_fase2_phone/v002/embeddings")):
        e = load_embeddings(d)
        idx = [i for i, x in enumerate(e.ids) if x in lab]
        m["loo_knn"][v] = loo_knn(e.X[idx], [lab[e.ids[i]] for i in idx])

    # 5. Phase 1 filter versus the class in the dedicated phase
    f1 = {r["image_id"]: ("Unassigned in phase 1" if r["status"] == "unassigned" else
                          "Ok in phase 1" if r["kept"] == "1" else "Discard in phase 1") for r in added["phone"]}
    rows_order = ["Ok in phase 1", "Discard in phase 1", "Unassigned in phase 1"]
    stack = [UNASSIGNED] + [c for c in CLASS_ORDER if c != "Fuco"]
    cross = {g: Counter(final[i] for i, s in f1.items() if s == g) for g in rows_order}
    m["fase1_vs_fase_telefono"] = {g: dict(c) for g, c in cross.items()}
    colour = {**COLOUR, UNASSIGNED: GREY}
    name = {UNASSIGNED: "Unassigned"}

    def stacked(ax, cats, counts):
        for k, cat in enumerate(cats):
            n = sum(counts[k].values())
            left = 0.0
            for c in stack:
                w = counts[k][c] / n if n else 0
                if w:
                    ax.barh(k, w, left=left, height=0.62, color=colour[c], edgecolor="white", linewidth=1.5)
                    if w >= 0.07:
                        ax.text(left + w / 2, k, f"{100 * w:.0f}%", ha="center", va="center", fontsize=8,
                                color="white" if c in ("Adulte", UNASSIGNED) else INK)
                left += w
        ax.set_yticks(range(len(cats)), cats)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=colour[c], label=name.get(c, english(c))) for c in stack],
                  loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=150)
    stacked(ax, [f"{g} ({sum(cross[g].values())})" for g in rows_order], [cross[g] for g in rows_order])
    ax.set_title("Same bees: phase-1 filter (camera model) vs class in the phone phase", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_fase1_vs_fase_telefono.png")
    plt.close(fig)

    # 6. Composition per photo
    photo = {}
    for r in pred:
        photo.setdefault(r["frame"], Counter())[final[r["image_id"]]] += 1
    order = sorted(photo)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    stacked(ax, [f"{p.replace('IMG_20261011_', '').replace('.jpg', '')} ({sum(photo[p].values())})" for p in order],
            [photo[p] for p in order])
    ax.set_title("Class of the cleaned bees in each phone photo", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_composizione_per_foto.png")
    plt.close(fig)

    # 7. UMAP map (embedding v002) with the phase 2 class colours
    coords_rows = read(AL / "results_fase2_phone/embedding_map/foto_telefono_v002/coords.csv")
    coords = np.array([[float(r["x"]), float(r["y"])] for r in coords_rows])
    plot(coords, [r["cluster"] for r in coords_rows], CLASS_ORDER, np.zeros(len(coords_rows), bool), OUT,
         "Phone phase - model v002")

    # 8. Crops per class
    for c in [c for c in CLASS_ORDER if c != "Fuco"] + [UNASSIGNED]:
        files = [PHONE_DIR / i.split("__", 1)[1] for i, v in final.items() if v == c]
        crop_grid(files, OUT / f"crops_{english(c).replace(' ', '_') if c != UNASSIGNED else 'Unassigned'}.png",
                  f"Phone phase, class {english(c) if c != UNASSIGNED else 'Unassigned'}")

    json.dump(m, open(OUT / "metrics.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: m[k] for k in ("seg_match_iou50", "fase1_filtro", "fase1_soglia_non_assegnate", "loo_knn",
                                        "fase1_vs_fase_telefono")}, indent=1, ensure_ascii=False, default=float)[:6000])
    print(f"Done: {OUT}")


if __name__ == "__main__":
    main()
