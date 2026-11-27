import os
import argparse
import cv2
import glob
import json
from collections import Counter
import multiprocessing
from typing import NamedTuple, Optional

import numpy as np

# ============================================================
# PARAMETRI DI PULIZIA (sanificazione)
# ============================================================
MIN_BLUR_SCORE = 50.0       # Sotto questa soglia l'immagine è troppo sfocata (punteggio a scala comune, vedi blur_score)
MIN_AREA_PIXELS = 1500      # Area minima della maschera (in pixel) per essere un'ape vera
MIN_FILL_RATIO = 0.30       # Un'ape riempie almeno il 30% del suo rettangolo minimo ruotato
MAX_ASPECT_RATIO = 3.0      # Rapporto lunghezza/larghezza massimo (evita frammenti tipo ali/zampe)
MIN_BBOX_SIDE = 25          # Lato corto minimo del rettangolo minimo ruotato (in pixel del crop originale)
# Sotto ~100 px di lunghezza i crop sono quasi solo pezzi d'ape (addomi nelle celle, teste,
# ali, api coperte) che passano area e aspect ratio; tra 100 e 120 px sono misti.
MIN_BEE_LENGTH = 105        # Lunghezza minima dell'ape (lato lungo del rettangolo ruotato, pixel del crop originale)

# Le api arrivano a scale molto diverse (da ~130 a ~490 px di lunghezza). I kernel
# morfologici qui sotto sono espressi per un'ape lunga REF_BEE_LENGTH px e vengono
# riscalati in proporzione alla lunghezza di ogni ape.
REF_BEE_LENGTH = 190        # Lunghezza di riferimento (picco della distribuzione delle api estratte)
KERNEL_SCALE_RANGE = (0.5, 3.0)
CLOSE_KERNEL = 15           # Chiusura dei buchi nel poligono grezzo
REFINE_DILATE_KERNEL = 11   # Dilatazione per non tagliare peli e zampe
OPEN_KERNEL = 25            # Apertura che spezza i ponti sottili tra api adiacenti
RECOVER_DILATE_KERNEL = 19  # Dilatazione che recupera zampe e antenne dopo l'apertura
SMOOTH_KERNEL = 31          # Blur+threshold che arrotonda i contorni
MAX_FRAGMENT_LOSS = 0.20    # Frazione massima di maschera persa tenendo solo il corpo principale
RAW_BORDER_MARGIN = 8       # Fascia (px) lungo il bordo del crop in cui cercare un'ape tagliata, maschere grezze
LEGACY_BORDER_MARGIN = 3    # Idem per le maschere legacy, già dilatate in estrazione
MAX_BORDER_PIXELS = 15      # Pixel di maschera tollerati in quella fascia

# ============================================================
# PARAMETRI DI NORMALIZZAZIONE
# ============================================================
TARGET_SIZE = 512           # Dimensione finale dell'immagine quadrata
TARGET_FILL_RATIO = 0.85    # Il lato lungo dell'ape occupa ~85% del frame dopo la normalizzazione
EDGE_FEATHER_PX = 12        # Pixel di sfumatura graduale ai bordi della maschera
# Testa in alto (bee_orientation.py). In run_cleaning è un passaggio a parte, a batch su
# GPU, sulle api già salvate: ha un suo registro nel manifesto e non invalida la pulizia.
ORIENT_HEAD_UP = True

# Da incrementare quando cambia la logica della pipeline (non solo i parametri):
# insieme ai parametri decide se il manifesto di una run precedente è ancora valido.
PIPELINE_VERSION = 4

# Esecuzione: non influenzano l'output, quindi non invalidano il manifesto
NUM_WORKERS = min(16, os.cpu_count() or 1)   # Processi in parallelo
MANIFEST_SAVE_EVERY = 500                    # Ogni quanti file salvare il manifesto

# Nome del file manifesto che traccia i file già processati (accettati e scartati)
CLEANING_MANIFEST_FILENAME = "_cleaning_manifest.json"


class CleaningResult(NamedTuple):
    """Esito della pulizia di un'ape. Se scartata, image/mask/matrix sono None e reason dice perché."""
    image: Optional[np.ndarray]    # TARGET_SIZE x TARGET_SIZE x 3, pronta da salvare
    mask: Optional[np.ndarray]     # maschera dell'ape nello stesso frame dell'immagine finale
    matrix: Optional[np.ndarray]   # affine 2x3 crop originale -> frame finale (per allineare altre maschere)
    reason: Optional[str]          # None se accettata, altrimenti il filtro che l'ha scartata


def cleaning_params():
    """Parametri che determinano l'output: se cambiano, il manifesto precedente non vale più."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "min_blur_score": MIN_BLUR_SCORE,
        "min_area_pixels": MIN_AREA_PIXELS,
        "min_fill_ratio": MIN_FILL_RATIO,
        "max_aspect_ratio": MAX_ASPECT_RATIO,
        "min_bbox_side": MIN_BBOX_SIDE,
        "min_bee_length": MIN_BEE_LENGTH,
        "ref_bee_length": REF_BEE_LENGTH,
        "kernel_scale_range": list(KERNEL_SCALE_RANGE),
        "close_kernel": CLOSE_KERNEL,
        "refine_dilate_kernel": REFINE_DILATE_KERNEL,
        "open_kernel": OPEN_KERNEL,
        "recover_dilate_kernel": RECOVER_DILATE_KERNEL,
        "smooth_kernel": SMOOTH_KERNEL,
        "max_fragment_loss": MAX_FRAGMENT_LOSS,
        "raw_border_margin": RAW_BORDER_MARGIN,
        "legacy_border_margin": LEGACY_BORDER_MARGIN,
        "max_border_pixels": MAX_BORDER_PIXELS,
        "target_size": TARGET_SIZE,
        "target_fill_ratio": TARGET_FILL_RATIO,
        "edge_feather_px": EDGE_FEATHER_PX,
    }




def new_cleaning_manifest():
    return {"params": cleaning_params(), "processed_files": {}, "discard_reasons": {}}


def load_cleaning_manifest(output_dir):
    """
    Carica il manifesto della pulizia (file già processati: accettati e scartati).

    Il manifesto vale solo per i parametri con cui è stato scritto: se sono cambiati
    (o è di una versione precedente che non li registrava) si riparte da zero, così
    tutti i file vengono riprocessati con i parametri attuali.
    """
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if manifest.get("params") == cleaning_params():
            manifest.setdefault("discard_reasons", {})
            return manifest
        print("⚠️  Parametri di pulizia cambiati rispetto al manifesto esistente: riprocesso tutti i file.")
    return new_cleaning_manifest()


def save_cleaning_manifest(output_dir, manifest):
    """Salva il manifesto della pulizia aggiornato."""
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_image_and_mask(path):
    """
    Carica un'immagine e la relativa maschera.

    Supporta due formati:
    - RGBA PNG (nuovo): il canale Alpha contiene la maschera grezza del poligono YOLO,
      preservata senza alcun processing dall'estrazione. Questo evita la perdita
      di informazione causata dalla ri-derivazione della maschera dai pixel.
    - RGB PNG/JPG (legacy): la maschera viene derivata dal threshold dei pixel non-neri.
      Compatibilità con immagini estratte dalla versione precedente dello script.

    Returns:
        img_bgr (H x W x 3), mask (H x W), is_raw_polygon (bool)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, False

    if img.ndim == 3 and img.shape[2] == 4:
        # RGBA: estrai maschera dal canale alpha (dati grezzi dal poligono YOLO)
        mask = img[:, :, 3]
        img_bgr = img[:, :, :3]
        return img_bgr, mask, True
    else:
        # RGB legacy: deriva maschera dal threshold (fallback per vecchie estrazioni)
        img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return img_bgr, mask, False


def largest_contour(mask):
    """Contorno esterno più grande della maschera, o None se è vuota."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def bee_length(mask):
    """Lunghezza dell'ape: lato lungo del rettangolo minimo ruotato (non dipende dall'orientamento)."""
    contour = largest_contour(mask)
    if contour is None:
        return 0.0
    return max(cv2.minAreaRect(contour)[1])


def kernel_scale(length):
    """Fattore con cui riscalare i kernel per un'ape di questa lunghezza."""
    return float(np.clip(length / REF_BEE_LENGTH, *KERNEL_SCALE_RANGE))


def scaled_size(base, scale):
    """Dimensione dispari di un kernel pensato per REF_BEE_LENGTH, riscalata."""
    return max(3, int(round(base * scale))) | 1


def scaled_ellipse(base, scale):
    size = scaled_size(base, scale)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def refine_raw_polygon_mask(mask, scale=1.0):
    """
    Raffina la maschera grezza del poligono YOLO.

    Applica le operazioni morfologiche che migliorano la continuità
    e la copertura della maschera. Queste operazioni venivano prima
    eseguite sia in extract_anomaly_dataset.py che qui, causando
    doppia elaborazione. Ora sono centralizzate solo qui.

    Pipeline:
    1. Chiusura morfologica → riempie buchi interni e fratture nel poligono
    2. Dilatazione → espande la maschera per coprire peli e zampe

    Questa funzione viene chiamata SOLO per maschere grezze (RGBA).
    Le immagini legacy (RGB) hanno già subìto queste operazioni
    durante l'estrazione e NON vengono ri-processate.

    scale: fattore dei kernel (vedi kernel_scale), 1.0 = ape lunga REF_BEE_LENGTH.
    """
    # Bordo nero temporaneo: senza, una chiusura con kernel più grande del padding del
    # crop lascerebbe la maschera attaccata al bordo (l'erosione non agisce da fuori).
    pad = scaled_size(CLOSE_KERNEL, scale)
    h, w = mask.shape[:2]
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 1. Chiusura morfologica: riempie buchi interni e unisce rientranze (fratture)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, scaled_ellipse(CLOSE_KERNEL, scale))
    
    # 2. Espandiamo la maschera per non tagliare peli e zampe
    mask = cv2.dilate(mask, scaled_ellipse(REFINE_DILATE_KERNEL, scale), iterations=1)
    
    return mask[pad:pad + h, pad:pad + w]


def blur_score(img_bgr, mask, length):
    """
    Nitidezza dell'ape: varianza del Laplaciano sui soli pixel della maschera,
    misurata a una scala comune.

    Alla risoluzione nativa il punteggio cala con la grandezza dell'ape (gli stessi
    dettagli sono spalmati su più pixel) e una soglia fissa scarta le api grandi
    anche quando sono a fuoco. Le api più lunghe di REF_BEE_LENGTH vengono quindi
    rimpicciolite a quella lunghezza prima della misura; le più corte restano alla
    risoluzione nativa, perché ingrandirle abbasserebbe il punteggio per la sola
    interpolazione.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, REF_BEE_LENGTH / max(length, 1.0))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    values = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
    return float(values.var()) if values.size else 0.0


def normalization_matrix(mask, target_size=TARGET_SIZE, target_fill=TARGET_FILL_RATIO):
    """
    Affine 2x3 che porta l'ape nel frame finale: asse maggiore verticale, lato lungo
    pari a target_fill * target_size, centrata nel quadrato target_size x target_size.

    Rotazione, scala e centratura stanno in un'unica trasformazione, così l'immagine
    viene interpolata una volta sola.

    Returns:
        (M, scale) oppure (None, None) se la maschera è vuota
    """
    contour = largest_contour(mask)
    if contour is None:
        return None, None

    # minAreaRect restituisce (centro, (larghezza, altezza), angolo)
    (cx, cy), (w_rect, h_rect), angle = cv2.minAreaRect(contour)
    long_side = max(w_rect, h_rect)
    if long_side <= 0:
        return None, None

    # Ruotando di `angle` il rettangolo diventa dritto, con w_rect lungo x e h_rect lungo y.
    # Se la larghezza > altezza l'asse maggiore è orizzontale: servono altri 90° per metterlo in verticale.
    rotation_angle = angle + 90 if w_rect > h_rect else angle
    scale = (target_size * target_fill) / long_side

    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, scale)
    # Il centro del rettangolo (cioè dell'ape) va al centro del frame finale
    M[0, 2] += target_size / 2 - cx
    M[1, 2] += target_size / 2 - cy
    return M, scale


def feather_edges(img_bgr, mask, feather_px=12):
    """
    Sfuma gradualmente i bordi della maschera per creare una transizione
    morbida tra l'ape e lo sfondo nero, eliminando il taglio netto.

    Il risultato è un'immagine dove i pixel ai bordi dell'ape "sfumano"
    dolcemente verso il nero, riducendo gli artefatti di bordo per PatchCore.

    Returns:
        img_feathered (stesse dimensioni)
    """
    # Creiamo una versione "erosa" della maschera (il nucleo sicuramente dentro l'ape)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
    inner_mask = cv2.erode(mask, erode_kernel, iterations=1)

    # Creiamo un gradiente morbido:
    # - Dentro (inner_mask) = 1.0 (completamente visibile)
    # - Bordo = sfumatura graduale da 1.0 a 0.0
    # - Fuori = 0.0 (completamente nero)
    # Per farlo, sfochiamo la maschera originale con un kernel grande
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (feather_px * 4 + 1, feather_px * 4 + 1), 0)

    # Forziamo l'interno a essere pieno (1.0) per non sfocare l'ape stessa
    alpha[inner_mask > 0] = 1.0

    # Applichiamo l'alpha all'immagine
    alpha_3c = np.stack([alpha] * 3, axis=-1)
    img_feathered = (img_bgr.astype(np.float32) * alpha_3c).astype(np.uint8)

    return img_feathered


def _discard(reason):
    return CleaningResult(None, None, None, reason)


def rotate_frame_180(result):
    """
    Ruota di 180° un'ape accettata: immagine, maschera e affine restano coerenti.

    cv2.rotate porta il pixel (x, y) in (S-1-x, S-1-y); la stessa mappa composta con
    l'affine fa sì che chi la riapplica (es. alle maschere varroa) resti allineato.
    """
    flip = np.array([[-1.0, 0.0, TARGET_SIZE - 1], [0.0, -1.0, TARGET_SIZE - 1], [0.0, 0.0, 1.0]])
    matrix = (flip @ np.vstack([result.matrix, [0.0, 0.0, 1.0]]))[:2]
    return CleaningResult(cv2.rotate(result.image, cv2.ROTATE_180), cv2.rotate(result.mask, cv2.ROTATE_180),
                          matrix, None)


def clean_bee(img_bgr, mask, is_raw_polygon, orienter=None):
    """
    Applica tutti i filtri di sanificazione e la normalizzazione su una singola ape.

    Args:
        img_bgr: immagine BGR (risoluzione originale del crop)
        mask: maschera binaria (dal canale alpha RGBA o da threshold legacy)
        is_raw_polygon: True se la maschera è il poligono grezzo YOLO (RGBA),
                        False se è già stata pre-processata (legacy RGB)
        orienter: bee_orientation.HeadOrienter per mettere la testa in alto; se None
                  l'ape resta con l'asse verticale ma la testa può essere in basso
                  (run_cleaning la orienta dopo, a batch)

    Returns:
        CleaningResult: immagine pulita e normalizzata, la sua maschera e l'affine
        usata; oppure solo il motivo dello scarto.
    """

    # I kernel morfologici seguono la grandezza dell'ape (misurata sulla maschera in ingresso)
    length = bee_length(mask)
    if length <= 0: return _discard("empty_mask")
    if length < MIN_BEE_LENGTH: return _discard("short")
    k = kernel_scale(length)

    # --- FILTRO 1: Ape tagliata ai bordi ---
    # Va fatto sulla maschera in ingresso: dopo il raffinamento la dilatazione, che cresce
    # con la grandezza dell'ape, porterebbe vicino al bordo anche api intere. Per i poligoni
    # grezzi il margine è più largo, come quando il controllo seguiva una dilatazione di 5 px.
    h_img, w_img = mask.shape[:2]
    border_margin = RAW_BORDER_MARGIN if is_raw_polygon else LEGACY_BORDER_MARGIN
    border_pixels = (cv2.countNonZero(mask[:border_margin, :]) + cv2.countNonZero(mask[h_img - border_margin:, :]) +
                     cv2.countNonZero(mask[:, :border_margin]) + cv2.countNonZero(mask[:, w_img - border_margin:]))
    if border_pixels > MAX_BORDER_PIXELS: return _discard("border")
    
    # --- RAFFINAMENTO MASCHERA (solo per poligoni grezzi RGBA) ---
    # Per le immagini RGBA (nuove), applica chiusura + dilatazione sulla maschera
    # grezza del poligono. Per le immagini RGB (legacy), queste operazioni erano
    # già state fatte in extract_anomaly_dataset.py, quindi le saltiamo.
    if is_raw_polygon:
        mask = refine_raw_polygon_mask(mask, k)

    # --- FILTRO 2: Spezza ponti sottili (thin bridges) ---
    # Kernel grande per staccare con forza api adiacenti o collegate
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, scaled_ellipse(OPEN_KERNEL, k))

    body_contour = largest_contour(opened_mask)
    if body_contour is None: return _discard("no_body")

    # --- FILTRO 3: Rimozione frammenti ---
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [body_contour], -1, 255, thickness=cv2.FILLED)

    # Dilatazione per recuperare zampe e antenne dopo l'apertura
    clean_mask = cv2.dilate(clean_mask, scaled_ellipse(RECOVER_DILATE_KERNEL, k), iterations=1)
    clean_mask = cv2.bitwise_and(clean_mask, mask)

    total_mask_area = cv2.countNonZero(mask)
    main_area = cv2.countNonZero(clean_mask)
    if total_mask_area > 0:
        if (1.0 - (main_area / total_mask_area)) > MAX_FRAGMENT_LOSS: return _discard("fragments")

    # Chiudiamo i buchi interni in modo definitivo riempiendo i contorni esterni
    contours_clean, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_clean:
        cv2.drawContours(clean_mask, contours_clean, -1, 255, thickness=cv2.FILLED)

    # --- FILTRO 4: Area minima ---
    area = cv2.countNonZero(clean_mask)
    if area < MIN_AREA_PIXELS: return _discard("area")

    # --- FILTRO 5: Aspect ratio ---
    # Misurato sul rettangolo minimo ruotato: con il bounding box dritto un'ape in
    # diagonale avrebbe rapporto ~1 e fill basso qualunque sia la sua forma.
    rect_w, rect_h = cv2.minAreaRect(largest_contour(clean_mask))[1]
    long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
    if short_side < MIN_BBOX_SIDE: return _discard("side")
    if long_side / short_side > MAX_ASPECT_RATIO: return _discard("aspect")

    # --- FILTRO 6: Fill ratio ---
    if area / (long_side * short_side) < MIN_FILL_RATIO: return _discard("fill")

    # --- FILTRO 7: Blur score ---
    if blur_score(img_bgr, clean_mask, long_side) < MIN_BLUR_SCORE: return _discard("blur")

    # --- SMOOTHING DEI CONTORNI ---
    # Un singolo passaggio di blur+threshold per arrotondare tutti i bordi
    # (prima veniva fatto sia in extract che qui — ora solo qui, una volta sola)
    smooth_size = scaled_size(SMOOTH_KERNEL, k)
    clean_mask = cv2.GaussianBlur(clean_mask, (smooth_size, smooth_size), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)

    # ============================================================
    # NORMALIZZAZIONE
    # ============================================================

    # 1. ORIENTAMENTO + SCALA: asse maggiore verticale, lato lungo a ~85% del frame 512x512.
    #    È l'UNICA interpolazione della pipeline: rotazione e resize sono una sola affine,
    #    applicata ai pixel originali del crop.
    M, scale = normalization_matrix(clean_mask)
    if M is None: return _discard("no_body")

    out_size = (TARGET_SIZE, TARGET_SIZE)
    cleaned_img = cv2.warpAffine(img_bgr, M, out_size,
                                 flags=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    clean_mask = cv2.warpAffine(clean_mask, M, out_size, flags=cv2.INTER_NEAREST, borderValue=0)
    cleaned_img = cv2.bitwise_and(cleaned_img, cleaned_img, mask=clean_mask)

    # Luminosità e contrasto non vengono toccati qui: sono già stati uniformati sull'intera
    # foto in extract_anomaly_dataset.py, e una correzione per singola ape li annullerebbe.

    # 2. SFUMATURA BORDI: transizione morbida ape → sfondo nero
    cleaned_img = feather_edges(cleaned_img, clean_mask, feather_px=EDGE_FEATHER_PX)

    result = CleaningResult(cleaned_img, clean_mask, M, None)

    # 3. TESTA IN ALTO: rotazione esatta di 180° se il classificatore vede la testa in basso
    if orienter is not None and orienter.scores([cleaned_img])[0] < 0:
        result = rotate_frame_180(result)
    return result


def clean_and_filter(img_bgr, mask, is_raw_polygon):
    """Come clean_bee, ma ritorna solo l'immagine finale (None se l'ape è scartata)."""
    return clean_bee(img_bgr, mask, is_raw_polygon).image


def _init_worker():
    # Le immagini sono piccole: i thread interni di OpenCV costano più di quanto rendono
    # e si pesterebbero i piedi tra processi.
    cv2.setNumThreads(1)


def _clean_file(job):
    """Pulisce un singolo file (eseguita nei processi worker). Ritorna (filename, status, reason, removed)."""
    path, out_path = job
    filename = os.path.basename(path)
    img_bgr, mask, is_raw_polygon = load_image_and_mask(path)
    if img_bgr is None:
        status, reason = "error", "unreadable"
    else:
        result = clean_bee(img_bgr, mask, is_raw_polygon)
        status, reason = ("accepted", None) if result.image is not None else ("discarded", result.reason)

    removed = False
    if status == "accepted":
        cv2.imwrite(out_path, result.image)
    elif os.path.exists(out_path):
        # Un'ape accettata da una run con parametri diversi non deve restare nel dataset
        os.remove(out_path)
        removed = True
    return filename, status, reason, removed


def orient_saved_bees(output_dir, manifest, device="auto", batch_size=256, workers=NUM_WORKERS):
    """
    Mette la testa in alto alle api accettate e già salvate in output_dir.

    Gira a batch su GPU dopo la pulizia, che lavora su CPU in più processi. Il passo è
    idempotente (un'ape già raddrizzata ha score positivo e non viene toccata), quindi
    un'interruzione a metà non può capovolgere due volte la stessa ape. Nel manifesto
    resta lo score di ogni ape prima del raddrizzamento: negativo = è stata ruotata,
    vicino a 0 = orientamento incerto.
    """
    from concurrent.futures import ThreadPoolExecutor
    from bee_orientation import HeadOrienter

    orienter = HeadOrienter(device=device)
    log = manifest.get("orientation")
    if not log or log.get("weights_id") != orienter.weights_id:
        # Pesi nuovi: si ricontrollano tutte le api (quelle già giuste restano come sono)
        log = {"weights_id": orienter.weights_id, "scores": {}}
    manifest["orientation"] = log

    todo = sorted(f for f, status in manifest["processed_files"].items()
                  if status == "accepted" and f not in log["scores"])
    print(f"Orientamento testa in alto su {orienter.device}: {len(todo)} api da controllare.")

    def rotate_and_save(item):
        path, img = item
        cv2.imwrite(path, cv2.rotate(img, cv2.ROTATE_180))

    rotated = 0
    with ThreadPoolExecutor(workers) as io:
        for start in range(0, len(todo), batch_size):
            names = todo[start:start + batch_size]
            paths = [os.path.join(output_dir, f) for f in names]
            loaded = [(f, p, img) for f, p, img in zip(names, paths, io.map(cv2.imread, paths)) if img is not None]
            scores = orienter.scores([img for _, _, img in loaded])
            to_rotate = [(p, img) for (_, p, img), score in zip(loaded, scores) if score < 0]
            list(io.map(rotate_and_save, to_rotate))
            rotated += len(to_rotate)
            for (f, _, _), score in zip(loaded, scores):
                log["scores"][f] = round(float(score), 3)
            if (start // batch_size) % 10 == 9:
                save_cleaning_manifest(output_dir, manifest)
                print(f"  Orientate {start + len(names)}/{len(todo)} api...")

    save_cleaning_manifest(output_dir, manifest)
    print(f"Ruotate di 180° {rotated} api su {len(todo)} controllate.")


def run_cleaning(input_dir, output_dir, workers=NUM_WORKERS, device="auto"):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))

    # Ignora il file manifesto dell'estrazione se presente nella cartella input
    img_paths = [p for p in img_paths if not os.path.basename(p).startswith("_")]

    # Carica il manifesto della pulizia per sapere quali file sono già stati processati
    manifest = load_cleaning_manifest(output_dir)

    # Skip dei file già processati (accettati o scartati) in una run precedente
    jobs = [(p, os.path.join(output_dir, os.path.basename(p))) for p in img_paths
            if os.path.basename(p) not in manifest["processed_files"]]

    total = len(img_paths)
    skipped = total - len(jobs)
    discarded = 0
    saved = 0
    removed = 0
    print(f"Trovate {total} immagini in '{input_dir}' ({skipped} già processate).")
    print(f"Inizio pulizia e normalizzazione con {workers} processi...")

    # Il tempo è dominato da lettura e scrittura su disco: più processi in parallelo
    # tengono occupato il disco invece di aspettarlo un file alla volta.
    # "spawn" e non "fork": un processo figlio creato per fork dopo che il padre ha usato
    # OpenCV eredita il suo pool di thread in uno stato inutilizzabile e si blocca.
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for done, (filename, status, reason, was_removed) in enumerate(
                pool.imap_unordered(_clean_file, jobs, chunksize=16), start=1):
            if status == "accepted":
                saved += 1
            else:
                discarded += 1
                manifest["discard_reasons"][filename] = reason
            removed += was_removed
            manifest["processed_files"][filename] = status

            # Salva il manifesto periodicamente per robustezza a interruzioni
            if done % MANIFEST_SAVE_EVERY == 0:
                save_cleaning_manifest(output_dir, manifest)
                print(f"  Processate {skipped + done}/{total} immagini...")

    # Salvataggio finale del manifesto
    save_cleaning_manifest(output_dir, manifest)

    print(f"\n📊 Risultato pulizia: {saved} api accettate, {discarded} scartate, {skipped} già processate su {total} totali.")
    reason_counts = Counter(manifest["discard_reasons"].values())
    if reason_counts:
        print("Motivi di scarto (tutte le run con questi parametri):")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<12} {count}")
    if removed:
        print(f"Rimosse da '{output_dir}' {removed} api accettate in precedenza e ora scartate.")

    if ORIENT_HEAD_UP:
        orient_saved_bees(output_dir, manifest, device=device, workers=workers)
    print(f"Api pulite e normalizzate salvate in '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulisce e normalizza le api estratte da extract_anomaly_dataset.py.")
    parser.add_argument("--raw", default="/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_single_bee_batch3+4_novarroa/raw_bees")
    parser.add_argument("--clean", default="/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_single_bee_batch3+4_novarroa/cleaned_bees")
    parser.add_argument("--device", default="auto", help="Device per l'orientamento (es. cuda:1); auto = GPU più libera")
    args = parser.parse_args()
    run_cleaning(args.raw, args.clean, device=args.device)
