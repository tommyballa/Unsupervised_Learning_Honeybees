import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

def main():
    # 1. Carica il modello base (che vogliamo migliorare con il fine-tuning)
    model = YOLO("yolo26s-seg.pt")
    print("Inizio il Fine-Tuning di YOLO con le etichette create da LangSAM...")
    
    # 2. Avvia l'addestramento
    # Ottimizzato per dataset piccolo (~90 immagini) con oggetti piccoli (api)
    results = model.train(
        data="data.yaml",             # Il file di configurazione del dataset
        epochs=150,                   # Più epoche per convergere meglio
        imgsz=1280,                   # Risoluzione alta, stessa della validazione
        batch=2,                      # Ridotto: con ~400 api/immagine la VRAM oscilla troppo
        device=0,                     # Usa la GPU primaria
        workers=2,                    # FONDAMENTALE SU WINDOWS
        project="runs/segment",       
        name="bee_model_finetuned_yolo26s_dominiuniti_splittestval",   
        max_det=1200,                  # Massimo numero di predizioni per immagine (molte api!)
        patience=20,                  # Early stopping con pazienza alta per non fermarsi troppo presto
        # --- Optimizer & Learning Rate ---
        optimizer='AdamW',            # Migliore convergenza su dataset piccoli
        lr0=0.001,                    # Learning rate iniziale
        lrf=0.01,                     # Learning rate finale (cosine decay: lr0 * lrf)
        cos_lr=True,                  # Cosine learning rate scheduler
        warmup_epochs=5,              # Warmup per stabilizzare l'inizio del training
        warmup_momentum=0.8,          # Momentum durante il warmup
        weight_decay=0.0005,          # Regolarizzazione L2
        # --- Augmentation (solo leggere, le pesanti causano OOM) ---
        mosaic=0.0,                   # Disattivato: riduce la risoluzione 4x
        hsv_h=0.015,                  # Variazione tonalità
        hsv_s=0.7,                    # Variazione saturazione
        hsv_v=0.4,                    # Variazione luminosità
        degrees=10.0,                 # Rotazione leggera
        translate=0.1,                # Traslazione
        scale=0.5,                    # Variazione scala (±50%)
        fliplr=0.5,                   # Flip orizzontale
        flipud=0.2,                   # Flip verticale
        # --- Varie ---
        label_smoothing=0.05,         # Leggero label smoothing per migliorare la generalizzazione
        overlap_mask=True,            # Gestisce maschere sovrapposte correttamente
    )
    
    print("\n Addestramento completato!")

if __name__ == "__main__":
    main()