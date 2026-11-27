import ast
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

def main():
    # 1. Load the base model (which we want to improve with fine-tuning)
    model = YOLO("yolo26s-seg.pt")
    print("Starting YOLO fine-tuning with the labels created by LangSAM...")
    
    # 2. Start training
    # Tuned for a small dataset (~300 images) with small objects (bees)
    params = dict(
        data="data.yaml",             # The dataset configuration file
        epochs=200,                   # More epochs for better convergence
        imgsz=1280,                   # High resolution, same as validation
        batch=2,                      # Reduced: with ~400 bees/image VRAM usage fluctuates too much
        device=1,                     # GPU 0 is busy with other jobs
        workers=8,
        cache=True,                   # Resized images in RAM: avoids decoding 24MP at every epoch
        project="runs/segment",       
        name="bee_model_finetuned_yolo26s_batch3+4",   
        max_det=1500,                  # Maximum number of predictions per image (many bees!)
        patience=40,                  # Early stopping with high patience so it does not stop too early
        # --- Optimizer & Learning Rate ---
        optimizer='AdamW',            # Better convergence on small datasets
        lr0=0.001,                    # Initial learning rate
        lrf=0.01,                     # Final learning rate (cosine decay: lr0 * lrf)
        cos_lr=True,                  # Cosine learning rate scheduler
        warmup_epochs=5,              # Warmup to stabilise the start of training
        warmup_momentum=0.8,          # Momentum during warmup
        weight_decay=0.0005,          # L2 regularisation
        # --- Augmentation (light only, heavy ones cause OOM) ---
        mosaic=0.0,                   # Disabled: reduces the resolution 4x
        hsv_h=0.015,                  # Hue variation
        hsv_s=0.7,                    # Saturation variation
        hsv_v=0.4,                    # Brightness variation
        degrees=10.0,                 # Slight rotation
        translate=0.1,                # Translation
        scale=0.5,                    # Scale variation (±50%)
        fliplr=0.5,                   # Horizontal flip
        flipud=0.2,                   # Vertical flip
        # --- Miscellaneous ---
        overlap_mask=True,            # Handles overlapping masks correctly
    )
    # Command-line overrides, e.g.: python train_yolo.py device=2 imgsz=1600 name=prova
    for arg in sys.argv[1:]:
        key, value = arg.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        params[key] = value
    results = model.train(**params)
    
    print("\n Training complete!")

if __name__ == "__main__":
    main()