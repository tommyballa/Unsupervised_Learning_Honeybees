"""Riassunto dei run in logs/supersimplenet_sweep: per ogni valutazione la testa cls e le migliori varianti."""
import glob, json, os
here = os.path.dirname(os.path.abspath(__file__))
for path in sorted(glob.glob(os.path.join(here, "*.json"))):
    d = json.load(open(path))
    print(f"== {os.path.basename(path)[:-5]}")
    for h in d["history"]:
        v = h["variants"]
        cls = next(r for r in v if r["score"] == "cls")
        ba = max(v, key=lambda r: r["img_auc"])
        bp = max(v, key=lambda r: r["pxl_ap"])
        print(f"  step {h['step']:6d} loss {h['loss']:.4f} | cls auc {cls['img_auc']:.4f} ap {cls['img_ap']:.4f} | "
              f"best auc {ba['img_auc']:.4f} ap {ba['img_ap']:.4f} (s{ba['sigma']:.0f} {ba['score']}) | "
              f"best pxl_ap {bp['pxl_ap']:.4f} (s{bp['sigma']:.0f})")
