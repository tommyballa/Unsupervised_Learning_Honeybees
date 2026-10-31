"""Confronto foto per foto tra YOLO img1376 (a 2048 px) e le etichette SAM 3 sulle 22 foto del telefono.

Per ogni foto: api di SAM 3, api del YOLO (alle soglie conf 0.10 della valutazione e 0.35 dell'estrazione),
abbinamenti maschera-maschera con IoU >= 0.5 (greedy per confidenza) e IoU medio degli abbinati.
Salva logs/phone/phone_compare.csv e, in visualizations/phone_yolo_vs_sam3/, un'immagine per foto:
SAM 3 a sinistra, YOLO (conf 0.35) a destra; in rosso le api che l'altro modello non ha.
Uso: python logs/phone/phone_compare.py <device> [cartella dati] [csv] [cartella immagini]
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

DEV = sys.argv[1] if len(sys.argv) > 1 else "0"
MODEL = "runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/phone_captured"
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "logs/phone/phone_compare.csv"
OUT_VIS = sys.argv[4] if len(sys.argv) > 4 else "visualizations/phone_yolo_vs_sam3"
SCALE = 0.5  # maschere confrontate a metà risoluzione


def gt_masks(label_path, w, h):
    masks = []
    for line in open(label_path):
        p = np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    return masks


def boxes_of(masks):
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        out.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, 0, 0))
    return np.array(out, float).reshape(-1, 4)


def match(pred, gt):
    """Greedy (pred già ordinate per confidenza): restituisce gli IoU degli abbinamenti e le gt abbinate."""
    pb, gb = boxes_of(pred), boxes_of(gt)
    used, ious = np.zeros(len(gt), bool), []
    for i, m in enumerate(pred):
        ix = (np.minimum(pb[i, 2], gb[:, 2]) >= np.maximum(pb[i, 0], gb[:, 0])) & \
             (np.minimum(pb[i, 3], gb[:, 3]) >= np.maximum(pb[i, 1], gb[:, 1])) & ~used
        best, bj = 0.0, -1
        for j in np.nonzero(ix)[0]:
            inter = np.logical_and(m, gt[j]).sum()
            iou = inter / (m.sum() + gt[j].sum() - inter + 1e-9)
            if iou > best:
                best, bj = iou, j
        if best >= 0.5:
            used[bj] = True
            ious.append(best)
    return np.array(ious), used


def draw(img, masks, ok):
    over = img.copy()
    for m, good in zip(masks, ok):
        color = (0, 200, 0) if good else (0, 0, 255)
        over[m] = (0.5 * over[m] + 0.5 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, color, 1)
    return over


def main():
    os.makedirs(OUT_VIS, exist_ok=True)
    model = YOLO(MODEL)
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "images", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)  # applica l'orientamento EXIF come la valutazione di ultralytics
        H, W = img.shape[:2]
        w, h = int(W * SCALE), int(H * SCALE)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gt = gt_masks(os.path.join(DATA, "labels", name + ".txt"), w, h)
        r = model.predict(path, imgsz=2048, conf=0.10, max_det=2500, device=DEV, retina_masks=False, verbose=False)[0]
        conf = r.boxes.conf.cpu().numpy() if r.masks is not None else np.zeros(0)
        order = np.argsort(-conf)
        pm = []
        if r.masks is not None:
            raw = r.masks.data.cpu().numpy()
            for i in order:
                pm.append(cv2.resize(raw[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))
        conf = conf[order]
        row = dict(foto=name, w=W, h=H, sam3=len(gt))
        for thr in (0.10, 0.35):
            sel = [m for m, c in zip(pm, conf) if c >= thr]
            ious, used = match(sel, gt)
            tag = f"{int(thr * 100):02d}"
            row[f"yolo_c{tag}"] = len(sel)
            row[f"tp_c{tag}"] = len(ious)
            row[f"prec_c{tag}"] = round(len(ious) / max(len(sel), 1), 3)
            row[f"rec_c{tag}"] = round(len(ious) / max(len(gt), 1), 3)
            row[f"iou_medio_c{tag}"] = round(float(ious.mean()), 3) if len(ious) else 0.0
            if thr == 0.35:
                _, pred_ok = match(gt, sel)  # quali api YOLO hanno una gemella in SAM 3
                left = draw(small, gt, used)
                right = draw(small, sel, pred_ok)
                for im, txt in ((left, f"SAM 3: {len(gt)}"), (right, f"YOLO conf>=0.35: {len(sel)}")):
                    cv2.putText(im, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
                cv2.imwrite(os.path.join(OUT_VIS, name + "_sam3_vs_yolo.jpg"), np.hstack([left, right]),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(row)
        print(row, flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("sam3", "yolo_c10", "tp_c10", "yolo_c35", "tp_c35")}
    print("Totale:", tot, "| conf 0.35: precisione %.3f richiamo %.3f" % (
        tot["tp_c35"] / tot["yolo_c35"], tot["tp_c35"] / tot["sam3"]))


if __name__ == "__main__":
    main()
