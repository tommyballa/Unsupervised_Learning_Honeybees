"""Copia di logs/ablation_sam3/abl_eval.py che legge i pesi da $ABL_RUNS invece che da runs/segment/runs/segment
(il disco di sistema era pieno, i run del freeze sono stati spostati su /mnt/disk1). Stesso protocollo.
Uso: ABL_RUNS=<cartella dei run> python abl_eval.py <device> <out.json> <nome_run>:<imgsz> ..."""
import os, sys, json, numpy as np
from ultralytics import YOLO
dev, out = int(sys.argv[1]), sys.argv[2]
R = os.environ["ABL_RUNS"] + "/"
MAX_DET = 2500
res = []
for spec in sys.argv[3:]:
    name, imgsz = spec.rsplit(":", 1)
    m = YOLO(R + name + "/weights/best.pt").val(data="data_sam3.yaml", split="test", imgsz=int(imgsz), batch=8, device=dev, conf=0.10, max_det=MAX_DET, plots=False, verbose=False, project=out + "_runs", name="v")
    res.append(dict(name=name, imgsz=int(imgsz), box_map=float(m.box.map), box_map50=float(m.box.map50), seg_map=float(m.seg.map), seg_map50=float(m.seg.map50), seg_map75=float(m.seg.map75), seg_p=float(np.mean(m.seg.p)), seg_r=float(np.mean(m.seg.r)), seg_f1=float(np.mean(m.seg.f1))))
    json.dump(res, open(out, "w"), indent=1)
