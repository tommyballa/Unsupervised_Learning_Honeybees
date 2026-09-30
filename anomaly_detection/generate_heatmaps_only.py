from pathlib import Path

import torch
from moviad.models.dinomaly.dinomaly import Dinomaly
from experiments.full_image.main_dinomaly_full_image import evaluate_and_visualize, TILE_SIZE, OUTPUT_DIR, SEED, set_seed

# checkpoint salvato da experiments/single_bee/main_dinomaly.py
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "single_bee" / "dinomaly.pth"

def main():
    set_seed(SEED)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Inizializzazione modello...")
    model = Dinomaly(
        encoder_name="vit_base_patch14_reg4_dinov2",
        target_layers=[2, 3, 4, 5],
        fuse_layer_encoder=[[0, 1, 2, 3]],
        img_size=(TILE_SIZE, TILE_SIZE),
    )
    model = model.to(device)

    print(f"Caricamento pesi da {CHECKPOINT}...")
    state_dict = torch.load(CHECKPOINT, map_location=device)
    
    # Gestisce sia il salvataggio dello state_dict diretto che quello dentro un dizionario (es. con epoche ecc.)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            # Nel caso in cui l'intero modello sia stato salvato invece dei soli pesi
            print(f"Errore nel caricare lo state_dict, provo a caricare come intero modello: {e}")
            model = torch.load(CHECKPOINT, map_location=device)
            model = model.to(device)

    print("Generazione heatmaps...")
    evaluate_and_visualize(model, device, output_dir=OUTPUT_DIR, window_size=TILE_SIZE)
    print("Fatto! Puoi controllare le immagini generate in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
