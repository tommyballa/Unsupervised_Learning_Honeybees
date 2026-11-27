"""Salva immagini casuali del test set batch3+4 con le api evidenziate dal modello scelto.
Uso: python predict_random_samples.py [n_immagini] [device]"""
from bee_paths import DATA_ROOT
import glob
import os
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4_img1440/weights/best.pt"
IMAGE_DIR = DATA_ROOT + "/processed/test_split_batch3+4/images"
OUT_DIR = "runs/segment/predict_batch3+4_img1440_samples"
IMGSZ = 2048        # stessa risoluzione usata nel test
CONF = 0.25
OUT_WIDTH = 2400    # larghezza delle immagini salvate
SEED = 42


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"  # "cpu" oppure indice GPU
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))
    random.seed(SEED)
    paths = random.sample(paths, min(n, len(paths)))
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS)
    rng = np.random.default_rng(SEED)

    for path in paths:
        img = cv2.imread(path)
        res = model.predict(source=img, imgsz=IMGSZ, conf=CONF, max_det=2000, device=device, verbose=False)[0]
        scale = OUT_WIDTH / img.shape[1]
        out = cv2.resize(img, (OUT_WIDTH, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        polys = [(p * scale).astype(np.int32) for p in (res.masks.xy if res.masks is not None else []) if len(p) >= 3]
        colors = rng.integers(60, 256, size=(len(polys), 3)).tolist()

        # Riempimento semitrasparente + contorno, un colore per ape
        overlay = out.copy()
        for poly, color in zip(polys, colors):
            cv2.fillPoly(overlay, [poly], color)
        out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
        for poly, color in zip(polys, colors):
            cv2.polylines(out, [poly], True, color, 2, cv2.LINE_AA)

        name = os.path.splitext(os.path.basename(path))[0] + "_pred.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, name), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name}: {len(polys)} api")


if __name__ == "__main__":
    main()
