import os
import numpy as np
from ultralytics import YOLO

def main():
    print(" Caricamento del modello...")

    model = YOLO("runs/segment/runs/segment/bee_model_finetuned_yolo26s_dominiuniti_splittestval/weights/best.pt")

    print("\n Inizio il Testing sul test set (calcolo metriche Precision, Recall, mAP, IoU, F1)...")


    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=2048,
        batch=16,
        device=0,
        conf=0.10,
        max_det=2000
    )
    
    print("\n Validazione completata! Controlla i risultati numerici qui sopra.")
    
    # --- Mask segmentation metrics ---
    print("\n--- Mask Segmentation Metrics ---")
    print(f"Mask mAP50:     {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95:  {metrics.seg.map:.4f}")
    print(f"Mask mAP75:     {metrics.seg.map75:.4f}")
    print(f"Mask Precision: {np.mean(metrics.seg.p):.4f}")
    print(f"Mask Recall:    {np.mean(metrics.seg.r):.4f}")
    print(f"Mask F1:        {np.mean(metrics.seg.f1):.4f}")

    # --- IoU totale ---
    # Calcolo IoU globale dalla relazione Precision/Recall (Jaccard index)
    # IoU = (P * R) / (P + R - P * R)  dove P e R sono la precision e recall medie
    p = np.mean(metrics.seg.p)
    r = np.mean(metrics.seg.r)
    denom = p + r - p * r
    iou_from_pr = (p * r) / denom if denom > 0 else 0.0
    print(f"Mask IoU (totale): {iou_from_pr:.4f}")

    # IoU micro-averaged dai conteggi aggregati (se disponibili)
    if hasattr(metrics.seg, 'image_metrics') and metrics.seg.image_metrics:
        total_tp = sum(im['tp'] for im in metrics.seg.image_metrics.values())
        total_fp = sum(im['fp'] for im in metrics.seg.image_metrics.values())
        total_fn = sum(im['fn'] for im in metrics.seg.image_metrics.values())
        total_denom = total_tp + total_fp + total_fn
        iou_micro = total_tp / total_denom if total_denom > 0 else 0.0
        print(f"Mask IoU (micro-avg): {iou_micro:.4f}")

    # --- Bounding box metrics for comparison ---
    print("\n--- Box Detection Metrics ---")
    print(f"Box mAP50:      {metrics.box.map50:.4f}")
    print(f"Box mAP50-95:   {metrics.box.map:.4f}")
    print(f"Box mAP75:      {metrics.box.map75:.4f}")
    print(f"Box Precision:  {np.mean(metrics.box.p):.4f}")
    print(f"Box Recall:     {np.mean(metrics.box.r):.4f}")
    print(f"Box F1:         {np.mean(metrics.box.f1):.4f}")

    print("\nI grafici (Curve Precision-Recall, Confusion Matrix, ecc.) sono stati salvati nella nuova cartella generata in 'runs/segment/val'")

if __name__ == "__main__":
    main()
