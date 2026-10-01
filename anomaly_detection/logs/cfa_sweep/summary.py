"""Riassunto degli sweep CFA: python logs/cfa_sweep/summary.py [tag ...]"""
import json, sys
from pathlib import Path

d = Path(__file__).parent
tags = sys.argv[1:] or sorted(p.stem for p in d.glob("*.json"))
for tag in tags:
    f = d / f"{tag}.json"
    if not f.exists():
        print(f"== {tag}: nessuna valutazione"); continue
    h = json.load(open(f))
    c = h["config"]
    print(f"== {tag}: " + " ".join(f"{k}={v}" for k, v in c.items() if k not in ("tag", "seed", "eval_zero", "eval_every")))
    for e in h["history"]:
        m = e["metrics"]
        line = (f"  step {e['step']:5d} ep {e['epoch']:.2f} loss {e['loss']:.4f} | auc {m['img_roc_auc']:.4f} "
                f"ap {m['img_avg_prec']:.4f} f1 {m['img_f1']:.4f} pxl_ap {m['pxl_avg_prec']:.4f}")
        if e["variants"]:
            bv = max(e["variants"], key=lambda r: r["img_auc"])
            bp = max(e["variants"], key=lambda r: r["pxl_ap128"])
            line += (f" | best var s={bv['sigma']} {bv['score']}: auc {bv['img_auc']:.4f} ap {bv['img_ap']:.4f}"
                     f" | best pxl s={bp['sigma']}: {bp['pxl_ap128']:.4f}")
        print(line)
