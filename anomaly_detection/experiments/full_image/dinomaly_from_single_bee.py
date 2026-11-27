"""
Full-image Dinomaly heatmaps using the weights trained on single bees.

Loads the best of experiments/single_bee/main_dinomaly.py (bottleneck +
decoder: the frozen DINOv2 encoder is the same and the decoder does not depend
on the tile size) into the full-image model and generates metrics and heatmaps
on the whole images in heatmaps/full_image/dinomaly_from_single_bee/.

For the Dinomaly trained directly on the whole images:
    python experiments/plot_heatmaps.py full_image dinomaly
"""

from pathlib import Path

import torch

from main_dinomaly_full_image import (
    SEED, TILE_SIZE, build_model, evaluate_and_visualize, load_model_state, set_seed,
)
from moviad.utilities.evaluation.checkpoints import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / "dinomaly"
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / "dinomaly_from_single_bee"


def main():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Initializing model...")
    model = build_model(device)

    state, info = load_checkpoint(CHECKPOINT_DIR)
    load_model_state(model, state)
    print(f"Single-bee weights loaded from {CHECKPOINT_DIR} (best: epoch {info.get('best_epoch')})")

    print("Generating heatmaps...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR, window_size=TILE_SIZE)
    print("Done! You can check the generated images in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
