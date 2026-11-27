import os
import sys
import numpy as np
from ultralytics import YOLO

def main():
    print(" Loading the model...")

    # Usage: python test_yolo.py [weights_path] [imgsz]
    weights = sys.argv[1] if len(sys.argv) > 1 else "runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4_img1440/weights/best.pt"
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    print(f" Weights: {weights} | imgsz: {imgsz}")
    model = YOLO(weights)

    print("\n Starting testing on the test set (computing Precision, Recall, mAP, IoU, F1)...")


    metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=imgsz,
        batch=16,
        device=1,                     # GPU 0 is busy with other jobs
        conf=0.10,
        max_det=2000
    )
    
    print("\n Validation complete! Check the numerical results above.")
    
    # --- Mask segmentation metrics ---
    print("\n--- Mask Segmentation Metrics ---")
    print(f"Mask mAP50:     {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95:  {metrics.seg.map:.4f}")
    print(f"Mask mAP75:     {metrics.seg.map75:.4f}")
    print(f"Mask Precision: {np.mean(metrics.seg.p):.4f}")
    print(f"Mask Recall:    {np.mean(metrics.seg.r):.4f}")
    print(f"Mask F1:        {np.mean(metrics.seg.f1):.4f}")

    # --- Overall IoU ---
    # Global IoU computed from the Precision/Recall relation (Jaccard index)
    # IoU = (P * R) / (P + R - P * R)  where P and R are the mean precision and recall
    p = np.mean(metrics.seg.p)
    r = np.mean(metrics.seg.r)
    denom = p + r - p * r
    iou_from_pr = (p * r) / denom if denom > 0 else 0.0
    print(f"Mask IoU (overall): {iou_from_pr:.4f}")

    # Micro-averaged IoU from the aggregated counts (if available)
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

    print("\nThe plots (Precision-Recall curves, Confusion Matrix, etc.) were saved in the new folder created under 'runs/segment/val'")

if __name__ == "__main__":
    main()
