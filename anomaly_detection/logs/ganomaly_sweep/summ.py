import json,sys
for f in sys.argv[1:]:
    try: d=json.load(open(f+".json"))
    except Exception: print(f,"n/a"); continue
    print("==",f)
    nm=lambda r:(str(r.get("err",""))+" s"+str(r.get("sigma",""))+" "+r["score"])
    for h in d["history"]:
        rows=h["rows"]
        bi=max(rows,key=lambda r:r["img_auc"]); bp=max(rows,key=lambda r:r.get("pxl_ap128",0))
        print(f"  ep{h['epoch']:2d} best_img: {nm(bi):20s} auc={bi['img_auc']:.4f} ap={bi['img_ap']:.4f} f1={bi['img_f1']:.4f} pxl_ap={bi.get('pxl_ap128',0):.4f} | best_pxl: {nm(bp):16s} auc={bp['img_auc']:.4f} pxl_ap={bp['pxl_ap128']:.4f}")
    for r in d["history"][-1]["rows"]:
        if r.get("err")=="chr" and r["score"] in ("max","topk10","topk100"): print("     ",r["err"],r["sigma"],r["score"],round(r["img_auc"],4),round(r["img_ap"],4),round(r["pxl_auc128"],4),round(r["pxl_ap128"],4))
