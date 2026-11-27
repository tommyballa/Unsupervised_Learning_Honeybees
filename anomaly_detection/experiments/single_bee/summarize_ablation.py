"""
Tabella e figure dell'ablation single-bee (numero di api di train, backbone).

Raccoglie i results.json dei run ufficiali (heatmaps/single_bee_val/<modello>,
= tutte le 102k api e backbone del recipe; patchcore_<backbone> per i backbone
PatchCore) e dell'ablation (heatmaps/ablation/<modello>/<tag>), più le misure
di measure_efficiency.py (logs/ablation/efficiency/), e scrive in
logs/ablation/summary/:

    ablation.csv       una riga per run
    train_size.pdf     AUROC immagine e AP pixel al variare di N (scala log)
    efficiency.pdf     AUROC immagine e AP pixel contro GFLOP per immagine, un punto per backbone

    python experiments/single_bee/summarize_ablation.py
"""

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "logs" / "ablation" / "summary"
EFF_DIR = REPO_ROOT / "logs" / "ablation" / "efficiency"
FULL_N = 102_303
MODELS = ["patchcore", "cfa", "dinomaly", "padim", "stfpm", "fastflow", "paste", "rd4ad"]
LABELS = {"patchcore": "PatchCore", "cfa": "CFA", "dinomaly": "Dinomaly", "padim": "PaDiM",
          "stfpm": "STFPM", "fastflow": "FastFlow", "paste": "PaSTe", "rd4ad": "RD4AD"}
# palette categoriale in ordine fisso (un colore per modello, mai per rango)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
SHORT_BB = {"wide_resnet50_2": "WRN50", "resnet18": "R18", "mobilenet_v2": "MNv2",
            "vit_base_patch14_reg4_dinov2": "ViT-B", "vit_small_patch14_reg4_dinov2": "ViT-S"}
BB_MARKER = {"wide_resnet50_2": "o", "vit_base_patch14_reg4_dinov2": "o",
             "resnet18": "s", "vit_small_patch14_reg4_dinov2": "s", "mobilenet_v2": "^"}
EFF_AUROC_MIN = 0.90
METRICS = ["img_roc_auc", "img_avg_prec", "img_f1", "pxl_avg_prec", "pxl_au_pro", "pxl_f1"]
EFF_KEYS = ["params_M", "size_MB", "gflops", "gpu_latency_ms", "gpu_throughput_bs32", "gpu_peak_MB_bs1", "cpu_latency_ms"]


def backbone_of(ckpt_dir):
    config = json.loads((ckpt_dir / "config.json").read_text()).get("config", {})
    return config.get("BACKBONE") or config.get("ENCODER_NAME")


def runs():
    """(modello, tag, backbone, N, dir heatmap, dir checkpoint) di tutti i run disponibili."""
    for model in MODELS:
        official = [(model, "default")] + ([("patchcore_" + b, b) for b in ("resnet18", "mobilenet_v2")]
                                           if model == "patchcore" else [])
        for name, tag in official:
            yield (model, tag, REPO_ROOT / "heatmaps" / "single_bee_val" / name,
                   REPO_ROOT / "checkpoints" / "single_bee_val" / name)
        for run_dir in sorted((REPO_ROOT / "heatmaps" / "ablation" / model).glob("*")):
            yield (model, run_dir.name, run_dir, REPO_ROOT / "checkpoints" / "ablation" / model / run_dir.name)


def collect():
    rows = []
    for model, tag, heat_dir, ckpt_dir in runs():
        if not (heat_dir / "results.json").exists():
            continue
        metrics = json.loads((heat_dir / "results.json").read_text())["metrics"]
        n = re.search(r"(?:^|_)n(\d+)$", tag)
        row = {"model": model, "tag": tag, "backbone": backbone_of(ckpt_dir),
               "train_n": int(n.group(1)) if n else FULL_N}
        row.update({k: metrics.get(k) for k in METRICS})
        eff = EFF_DIR / f"{model}_{tag}.json"
        if eff.exists():
            eff = json.loads(eff.read_text())
            row.update({k: eff.get(k) for k in EFF_KEYS})
        rows.append(row)
    return rows


def write_csv(rows):
    keys = ["model", "tag", "backbone", "train_n"] + METRICS + EFF_KEYS
    with open(OUT_DIR / "ablation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


def style(ax):
    ax.grid(True, color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a8980")
    ax.tick_params(colors="#55544e", labelsize=8)


def plot_train_size(rows, default_bb):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ["img_roc_auc", "pxl_avg_prec"], ["Image AUROC", "Pixel AP"]):
        for i, model in enumerate(MODELS):
            pts = sorted((r["train_n"], r[metric]) for r in rows
                         if r["model"] == model and r["backbone"] == default_bb.get(model) and r[metric] is not None)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=6, linewidth=2, label=LABELS[model])
        ax.set_xscale("log")
        ax.set_xlabel("training bees (N, log scale)", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", color="#1f1e1b")
        style(ax)
    axes[1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_size.pdf")
    fig.savefig(OUT_DIR / "train_size.png", dpi=150)
    plt.close(fig)


def plot_efficiency(rows):
    # GFLOP e non latenza: le GPU erano condivise con altri job, i tempi sono solo indicativi
    full = [r for r in rows if r["train_n"] == FULL_N and r.get("gflops")]
    if not full:
        return
    # colore = modello, marker = taglia del backbone (niente etichette di testo, che si sovrappongono);
    # sull'AUROC l'asse parte da EFF_AUROC_MIN e i punti sotto sono disegnati sul bordo con il valore
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, metric, title, ylo in zip(axes, ["img_roc_auc", "pxl_avg_prec"], ["Image AUROC", "Pixel AP"],
                                      [EFF_AUROC_MIN, None]):
        below = []
        for i, model in enumerate(MODELS):
            pts = sorted((r["gflops"], r[metric], r["backbone"]) for r in full if r["model"] == model)
            if not pts:
                continue
            xs, ys, bbs = zip(*pts)
            shown = [max(y, ylo) if ylo else y for y in ys]
            ax.plot(xs, shown, color=COLORS[i], linewidth=1.5, alpha=0.6, zorder=2)
            for x, y, s, bb in zip(xs, ys, shown, bbs):
                ax.scatter(x, s, color=COLORS[i], marker=BB_MARKER[bb], s=80, edgecolors="white",
                           linewidths=1.2, zorder=3, clip_on=False)
                if s != y:
                    below.append(f"{LABELS[model]} {SHORT_BB[bb]} {y:.3f}")
        if below:
            ax.text(1.0, 1.015, f"off scale, drawn at {ylo:.2f}: " + ", ".join(below), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="#55544e")
        ax.set_xscale("log")
        if ylo:
            ax.set_ylim(ylo, 1.002)
        ax.set_xlabel("GFLOPs per image (log scale)", fontsize=10)
        ax.set_title(title, fontsize=11, loc="left", color="#1f1e1b")
        style(ax)
        ax.tick_params(labelsize=9)
    model_handles = [Line2D([], [], color=COLORS[i], linewidth=2.5, label=LABELS[m])
                     for i, m in enumerate(MODELS) if any(r["model"] == m for r in full)]
    bb_handles = [Line2D([], [], color="#55544e", marker=mk, linestyle="", markersize=8, label=lab)
                  for lab, mk in [("WRN50 / ViT-B", "o"), ("ResNet-18 / ViT-S", "s"), ("MobileNetV2", "^")]]
    fig.legend(handles=model_handles, fontsize=9, frameon=False, ncol=8, title="Model", title_fontsize=9,
               loc="lower center", bbox_to_anchor=(0.36, 0.0))
    fig.legend(handles=bb_handles, fontsize=9, frameon=False, ncol=3, title="Backbone", title_fontsize=9,
               loc="lower center", bbox_to_anchor=(0.84, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT_DIR / "efficiency.pdf")
    fig.savefig(OUT_DIR / "efficiency.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_csv(rows)
    default_bb = {r["model"]: r["backbone"] for r in rows if r["tag"] == "default"}
    plot_train_size(rows, default_bb)
    plot_efficiency(rows)
    for r in rows:
        print(f"{r['model']:>10s} {r['tag']:>32s} {str(r['backbone']):>30s} N={r['train_n']:>6d} "
              f"AUROC={r['img_roc_auc']:.4f} pxlAP={r['pxl_avg_prec']:.4f}"
              + (f" lat={r['gpu_latency_ms']:.1f}ms" if r.get("gpu_latency_ms") else ""))
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
