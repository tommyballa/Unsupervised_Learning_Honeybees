"""
Heatmap full-image di Dinomaly con i pesi addestrati sulle singole api.

Carica la best di experiments/single_bee/main_dinomaly.py (bottleneck +
decoder: l'encoder DINOv2 congelato è lo stesso e il decoder non dipende dalla
dimensione del tile) nel modello full-image e genera metriche e heatmap sulle
immagini intere in heatmaps/full_image/dinomaly_from_single_bee/.

Per il Dinomaly addestrato direttamente sulle immagini intere:
    python experiments/plot_heatmaps.py full_image dinomaly
"""

from pathlib import Path

import torch

from experiments.full_image.main_dinomaly_full_image import (
    SEED, TILE_SIZE, build_model, evaluate_and_visualize, load_model_state, set_seed,
)
from moviad.utilities.evaluation.checkpoints import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "single_bee" / "dinomaly"
OUTPUT_DIR = REPO_ROOT / "heatmaps" / "full_image" / "dinomaly_from_single_bee"


def main():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Inizializzazione modello...")
    model = build_model(device)

    state, info = load_checkpoint(CHECKPOINT_DIR)
    load_model_state(model, state)
    print(f"Pesi single-bee caricati da {CHECKPOINT_DIR} (best: epoca {info.get('best_epoch')})")

    print("Generazione heatmaps...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR, window_size=TILE_SIZE)
    print("Fatto! Puoi controllare le immagini generate in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
