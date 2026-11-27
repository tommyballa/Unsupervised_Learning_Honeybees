from bee_paths import DATA_ROOT
import os
import argparse
import cv2
import glob
import json
import numpy as np
from ultralytics import YOLO

# Nome del file manifesto che traccia le immagini sorgente già processate
MANIFEST_FILENAME = "_extraction_manifest.json"

# ============================================================
# NORMALIZZAZIONE PER FOTO (luminosità e contrasto)
# ============================================================
# Prima del ritaglio la foto viene portata a luminosità e contrasto comuni con una
# trasformazione lineare del canale L (LAB): un solo guadagno e un solo offset per
# tutta la foto, quindi le api della stessa foto restano confrontabili tra loro.
# Le statistiche sono misurate sui soli pixel delle api, perché lo sfondo (favo,
# legno, telaino) cambia da foto a foto. I target sono le mediane misurate su 90
# foto di train/test batch3+4 (media L 72-139, deviazione 33-51 prima della correzione).
NORMALIZE_PHOTO = True
TARGET_L_MEAN = 110.0       # Luminosità media delle api dopo la normalizzazione (L in 0-255)
TARGET_L_STD = 42.0         # Contrasto (deviazione standard di L) delle api dopo la normalizzazione
GAIN_RANGE = (0.6, 1.8)     # Limiti del guadagno, per non amplificare il rumore nelle foto piatte
MIN_BEE_PIXELS = 5000       # Sotto questa soglia le statistiche si misurano sull'intera foto


def load_manifest(out_dir):
    """Carica il manifesto delle immagini sorgente già processate."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {"processed_sources": {}, "next_bee_id": 0}


def save_manifest(out_dir, manifest):
    """Salva il manifesto aggiornato su disco."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def photo_normalization_params():
    """Parametri che determinano i pixel salvati: api estratte con parametri diversi non vanno mescolate."""
    if not NORMALIZE_PHOTO:
        return {"enabled": False}
    return {"enabled": True, "target_l_mean": TARGET_L_MEAN, "target_l_std": TARGET_L_STD,
            "gain_range": list(GAIN_RANGE), "min_bee_pixels": MIN_BEE_PIXELS}


def normalize_photo(img_bgr, bee_mask):
    """
    Porta luminosità e contrasto della foto ai valori target, misurandoli sui pixel delle api.

    Returns:
        (immagine normalizzata, guadagno, offset) con L' = L * guadagno + offset
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    use_mask = bee_mask if cv2.countNonZero(bee_mask) >= MIN_BEE_PIXELS else None
    mean, std = cv2.meanStdDev(l_ch, mask=use_mask)
    mean, std = float(mean[0, 0]), float(std[0, 0])

    gain = TARGET_L_STD / std if std > 1e-6 else 1.0
    gain = max(GAIN_RANGE[0], min(gain, GAIN_RANGE[1]))
    offset = TARGET_L_MEAN - mean * gain

    lab[:, :, 0] = cv2.convertScaleAbs(l_ch, alpha=gain, beta=offset)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), gain, offset


def parse_args():
    parser = argparse.ArgumentParser(description="Estrae le singole api segmentate da YOLO come PNG RGBA.")
    parser.add_argument("--model", default="runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4_img1440/weights/best.pt")
    # Cartella da cui estrarre le api
    parser.add_argument("--source", default=DATA_ROOT + "/processed/test_batch3+4/images")
    # Output: cartella con TUTTE le api estratte grezze (senza filtri)
    parser.add_argument("--out", default=DATA_ROOT + "/processed/test_single_bee_batch3+4_novarroa/raw_bees")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None, help="Es. 0 oppure cpu; default: scelta automatica di ultralytics")
    return parser.parse_args()


def main():
    args = parse_args()
    print(" Caricamento del modello YOLO...")
    model = YOLO(args.model)
    source_dir = args.source
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    # Carica il manifesto per sapere quali immagini sorgente sono già state processate
    manifest = load_manifest(out_dir)

    # Le api già estratte devono avere la stessa normalizzazione di quelle nuove
    norm_params = photo_normalization_params()
    previous_params = manifest.get("photo_normalization", {"enabled": False})
    if manifest["processed_sources"] and previous_params != norm_params:
        raise SystemExit(
            f"'{out_dir}' contiene api estratte con una normalizzazione diversa ({previous_params}).\n"
            f"Usa una cartella di output nuova o svuota quella esistente per non mescolarle.")
    manifest["photo_normalization"] = norm_params
    
    print(f"Eseguendo l'estrazione delle api da: {source_dir}")
    
    image_paths = (
        glob.glob(os.path.join(source_dir, "*.[jJ][pP][gG]")) +
        glob.glob(os.path.join(source_dir, "*.[pP][nN][gG]"))
    )
    
    bee_counter = manifest["next_bee_id"]
    skipped = 0
    extracted = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Skip se questa immagine sorgente è già stata processata in una run precedente
        if img_name in manifest["processed_sources"]:
            skipped += 1
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Facciamo inferenza alla massima risoluzione. retina_masks=True migliora enormemente i bordi.
        results = model.predict(source=img, save=False, show=False, conf=args.conf, imgsz=args.imgsz,
                                max_det=2000, device=args.device, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            manifest["processed_sources"][img_name] = {"bees": 0}
            save_manifest(out_dir, manifest)
            continue
            
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        # Normalizzazione per foto: l'inferenza è già stata fatta sulla foto originale,
        # da qui in poi i crop vengono presi dalla foto normalizzata.
        photo_stats = {}
        if NORMALIZE_PHOTO:
            all_bees_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for polygon in res.masks.xy:
                if len(polygon) > 0:
                    cv2.fillPoly(all_bees_mask, [np.array(polygon, np.int32).reshape((-1, 1, 2))], 255)
            img, gain, offset = normalize_photo(img, all_bees_mask)
            photo_stats = {"gain": round(gain, 4), "offset": round(offset, 2)}
        
        bees_from_this_image = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            h, w = img.shape[:2]
            
            # Crea una maschera nera delle dimensioni dell'immagine originale
            blank_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Prendi le coordinate del poligono di questa specifica ape
            polygon = res.masks.xy[i]
            if len(polygon) == 0:
                continue
            
            # Disegna il poligono riempito di bianco sulla maschera nera
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(blank_mask, [pts], 255)
            
            # ---------------------------------------------------------------
            # NOTA: le operazioni di raffinamento della maschera (chiusura
            # morfologica, dilatazione, GaussianBlur+threshold) NON vengono
            # più eseguite qui. Sono centralizzate in dataset_cleaning.py
            # per evitare duplicazioni e preservare la qualità originale.
            # ---------------------------------------------------------------
            
            # Crop con padding generoso attorno al bounding box
            padding = 20
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(w, x2 + padding)
            y2_p = min(h, y2 + padding)
            
            # Ritagliamo sia la foto che la maschera usando le coordinate maggiorate
            img_crop = img[y1_p:y2_p, x1_p:x2_p].copy()
            mask_crop = blank_mask[y1_p:y2_p, x1_p:x2_p]
            
            # ---------------------------------------------------------------
            # Salva come RGBA PNG: i canali BGR contengono l'immagine originale
            # non modificata, il canale Alpha contiene la maschera grezza del
            # poligono YOLO. Questo preserva sia i pixel originali sia la
            # maschera senza alcuna perdita di qualità.
            #
            # NON facciamo più il resize a 512x512 qui: il resize singolo
            # avviene in dataset_cleaning.py dopo tutti i filtri, evitando
            # il doppio resize che degradava l'immagine.
            # ---------------------------------------------------------------
            b_ch, g_ch, r_ch = cv2.split(img_crop)
            rgba = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
            
            # Salva il risultato con la confidenza nel nome per debug (PNG lossless con zero compressione per massima fedeltà)
            bee_filename = f"bee_{bee_counter:05d}_conf{confs[i]:.2f}.png"
            cv2.imwrite(os.path.join(out_dir, bee_filename), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            bee_counter += 1
            bees_from_this_image += 1
            extracted += 1
        
        # Aggiorna il manifesto dopo ogni immagine sorgente per robustezza a interruzioni
        manifest["processed_sources"][img_name] = {"bees": bees_from_this_image, **photo_stats}
        manifest["next_bee_id"] = bee_counter
        save_manifest(out_dir, manifest)
            
    print(f"\n✅ Estrazione completata! Estratte {extracted} nuove api ({skipped} immagini sorgente già processate, saltate).")
    print(f"Totale api nella cartella: {bee_counter}")
    print(f"Api grezze salvate in '{out_dir}'.")
    print("Ora lancia dataset_cleaning.py per pulire e normalizzare il dataset.")

if __name__ == "__main__":
    main()
