import cv2
import numpy as np
import random
import os
import glob

# --- Configurazione ---
IMAGES_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_batch3+4/images"
LABELS_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_batch3+4/labels"
VARROA_PATHS = [
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha.png",
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha2.png",
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha3.png",
]
OUTPUT_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test_batch3+4/synthetic_varroa_output"

# Seed per rendere riproducibile la generazione (None = casuale a ogni esecuzione)
SEED = 42

# Quante varianti sintetiche generare per ogni immagine
NUM_VARIANTS = 1

# Rapporto dimensionale Varroa/Ape: la Varroa destructor è ~1.6mm di lunghezza,
# il torace dell'ape è ~4-5mm, ma in foto il corpo intero occupa molto di più.
# Usiamo il lato più corto della bounding box del poligono come riferimento per il torace.
VARROA_BEE_RATIO_MIN = 0.20
VARROA_BEE_RATIO_MAX = 0.24

# Larghezza (in pixel) a cui viene normalizzato il corpo di tutte le varroa sorgente
VARROA_REF_BODY_WIDTH = 128

# Quante api infettare per immagine
NUM_BEES_TO_INFECT_MIN = 20
NUM_BEES_TO_INFECT_MAX = 40

# Quante varroa inserire per ogni ape selezionata (da 1 a 2)
VARROA_PER_BEE_MIN = 1
VARROA_PER_BEE_MAX = 2

# --- Fusione della varroa con l'ape (0 = varroa originale, 1 = adattamento totale) ---
BLEND_LUMINANCE = 0.60   # quanto la luminosità media segue quella dell'ape
BLEND_CONTRAST = 0.45    # quanto il contrasto segue quello dell'ape
BLEND_CHROMA = 0.45      # quanto il colore vira verso quello dell'ape
BLEND_SHADING = 0.60     # quanto la varroa eredita luci e ombre locali dell'ape
# Sfocatura (sigma in pixel) per adattare la nitidezza a quella della foto:
# minima sempre applicata, massima in proporzione alla grandezza della varroa
MIN_BLUR_SIGMA = 0.6
MAX_BLUR_RATIO = 0.07
# Tetto alla grana aggiunta (dev. std), stimata dalla foto
MAX_NOISE_SIGMA = 6.0
# Opacità delle ombre
CONTACT_SHADOW_OPACITY = 0.25
DROP_SHADOW_OPACITY = 0.20


def resize_rgba(image_rgba, scale):
    """
    Ridimensiona un'immagine RGBA lavorando in alpha premoltiplicato, così il
    colore (nero) dei pixel trasparenti non sporca i bordi della varroa.
    """
    h, w = image_rgba.shape[:2]
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # Usa INTER_AREA per downscale, INTER_CUBIC per upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(_premultiply(image_rgba), new_size, interpolation=interp)
    return _unpremultiply(resized)


def _premultiply(image_rgba):
    img = image_rgba.astype(np.float32)
    img[:, :, :3] *= img[:, :, 3:4] / 255.0
    return img


def _unpremultiply(image_rgba_float):
    img = image_rgba_float.copy()
    img[:, :, :3] /= np.maximum(img[:, :, 3:4] / 255.0, 1e-6)
    return np.clip(img, 0, 255).astype(np.uint8)


def _clean_varroa_alpha(alpha):
    """
    Ripulisce la maschera di una varroa sorgente: toglie la frangia chiara
    lasciata dallo scontorno sul bordo, rimuove peli e zampe sottili (che
    gonfierebbero la bounding box e sporcherebbero la GT mask) e chiude i buchi.
    """
    w = alpha.shape[1]
    _, alpha = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

    fringe = max(1, round(w * 0.02))
    alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=fringe)

    k = max(3, int(w * 0.05)) | 1
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(alpha)
    body = np.zeros_like(alpha)
    cv2.drawContours(body, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return body


def load_varroa_sources():
    """Carica tutte le immagini RGBA delle varroa.

    Ogni varroa viene ripulita, ritagliata sul corpo e normalizzata alla
    larghezza VARROA_REF_BODY_WIDTH: così le differenze di risoluzione e di
    margine trasparente tra i vari PNG non influenzano il dimensionamento
    finale sulle api.
    """
    normalized = []
    for vp in VARROA_PATHS:
        v = cv2.imread(vp, cv2.IMREAD_UNCHANGED)
        if v is None:
            raise FileNotFoundError(f"Impossibile caricare l'immagine della varroa: {vp}")
        if v.ndim != 3 or v.shape[2] != 4:
            raise ValueError(f"L'immagine della Varroa deve avere un canale Alpha (PNG 32-bit): {vp}")

        alpha = _clean_varroa_alpha(v[:, :, 3])
        if cv2.countNonZero(alpha) == 0:
            raise ValueError(f"L'immagine della Varroa non ha pixel opachi: {vp}")

        # Ritaglia sul corpo
        x, y, w, h = cv2.boundingRect(alpha)
        orig_alpha = v[y:y + h, x:x + w, 3]
        alpha = alpha[y:y + h, x:x + w]
        rgb = np.ascontiguousarray(v[y:y + h, x:x + w, :3])

        # I buchi chiusi nella maschera erano trasparenti nel PNG: riempili
        # con i colori circostanti
        holes = cv2.bitwise_and(alpha, cv2.bitwise_not(cv2.threshold(orig_alpha, 128, 255, cv2.THRESH_BINARY)[1]))
        if cv2.countNonZero(holes) > 0:
            rgb = cv2.inpaint(rgb, holes, 3, cv2.INPAINT_TELEA)

        normalized.append(resize_rgba(np.dstack([rgb, alpha]), VARROA_REF_BODY_WIDTH / w))

    return normalized


def parse_yolo_segmentation_labels(label_path, img_w, img_h):
    """
    Legge le label YOLO segmentation: ogni riga è
       class_id x1 y1 x2 y2 ... xN yN
    dove le coordinate sono normalizzate [0,1].

    Restituisce una lista di dizionari con:
      - 'polygon': np.array di punti (N,2) in pixel
      - 'bbox': (x, y, w, h) bounding box in pixel
      - 'centroid': (cx, cy) centro del poligono in pixel
    """
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + almeno 3 punti (6 valori)
                continue

            # Primo valore è il class_id, il resto sono coppie x,y
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue  # Numero dispari di coordinate, skip

            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                points.append([px, py])

            polygon = np.array(points, dtype=np.float32)

            # Bounding box dal poligono
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            # Centroide
            cx = polygon[:, 0].mean()
            cy = polygon[:, 1].mean()

            detections.append({
                "polygon": polygon,
                "bbox": bbox,
                "centroid": (int(cx), int(cy)),
            })

    return detections


def rotate_image_rgba(image_rgba, angle):
    """
    Ruota un'immagine RGBA di un angolo arbitrario, espandendo il bounding box
    per non tagliare i bordi. I pixel aggiunti dall'espansione sono trasparenti.
    """
    h, w = image_rgba.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Calcola le nuove dimensioni per contenere l'immagine ruotata
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Aggiusta la matrice di trasformazione per il nuovo centro
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Alpha premoltiplicato: l'interpolazione non mescola il nero dei pixel
    # trasparenti nei bordi della varroa
    rotated = cv2.warpAffine(
        _premultiply(image_rgba), M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return _unpremultiply(rotated)


def largest_component(mask):
    """
    Tiene solo la componente connessa più grande della maschera: dopo resize e
    rotazione zampe e peli sottili possono staccarsi dal corpo e finirebbero
    nella GT mask come frammenti isolati.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    biggest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def bleed_colors(rgb, mask):
    """
    Estende i colori della varroa oltre il bordo della maschera, così la
    sfumatura dell'alpha non mescola il nero dei pixel trasparenti.
    """
    return cv2.inpaint(rgb, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)


def adapt_color_to_region(varroa_rgb, target_region, alpha_mask, bg_patch=None):
    """
    Adatta colore, luminosità e contrasto della varroa alla foto dell'ape.
    1. Match del contrasto (deviazione standard) oltre che della media luminosa
    2. Blend della crominanza per assorbire i riflessi dell'ambiente
    3. Trasferimento dell'illuminazione locale dell'ape (zone in luce / in ombra)

    bg_patch: porzione dell'immagine allineata pixel per pixel con varroa_rgb
    (stessa dimensione); se None il punto 3 viene saltato.
    Nitidezza e grana vengono adattate dopo, in match_sharpness_and_noise.
    """
    if target_region.size == 0 or varroa_rgb.size == 0:
        return varroa_rgb

    # Maschera dei pixel opachi della varroa
    opaque = alpha_mask > 128

    if opaque.sum() < 10:
        return varroa_rgb

    # Converti in spazio colore percettivo LAB
    varroa_lab = cv2.cvtColor(varroa_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Estrai i canali di luminosità (L)
    v_l = varroa_lab[:, :, 0][opaque]
    t_l = target_lab[:, :, 0]

    v_mean_l, v_std_l = v_l.mean(), v_l.std() + 1e-5
    t_mean_l, t_std_l = t_l.mean(), t_l.std() + 1e-5

    # 1. Match del Contrasto e Luminosità
    target_contrast = (v_std_l * (1.0 - BLEND_CONTRAST) + t_std_l * BLEND_CONTRAST) / v_std_l
    l_shift = (t_mean_l - v_mean_l) * BLEND_LUMINANCE
    new_l = (varroa_lab[:, :, 0] - v_mean_l) * target_contrast + v_mean_l + l_shift

    # 3. Illuminazione locale: la varroa segue le variazioni lente di luce
    # dell'ape sottostante invece di avere una luminosità uniforme
    if bg_patch is not None and bg_patch.shape[:2] == varroa_rgb.shape[:2]:
        bg_l = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        sigma = max(2.0, np.sqrt(opaque.sum()) * 0.25)
        bg_low = cv2.GaussianBlur(bg_l, (0, 0), sigma)
        shading = bg_low / max(bg_low[opaque].mean(), 1.0)
        new_l = new_l * (1.0 + BLEND_SHADING * (shading - 1.0))

    # Applica trasformazione al canale L
    varroa_lab[:, :, 0] = np.where(opaque, np.clip(new_l, 0, 255), varroa_lab[:, :, 0])

    # 2. Match della Crominanza
    for ch in [1, 2]:
        v_ch = varroa_lab[:, :, ch][opaque]
        t_ch = target_lab[:, :, ch]
        # Shift parziale per preservare il colore tipico del parassita
        shift = (t_ch.mean() - v_ch.mean()) * BLEND_CHROMA
        varroa_lab[:, :, ch] = np.where(
            opaque,
            np.clip(varroa_lab[:, :, ch] + shift, 0, 255),
            varroa_lab[:, :, ch]
        )

    # Converti di nuovo in BGR
    return cv2.cvtColor(varroa_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _sharpness(gray, mask=None):
    """Misura di nitidezza: 90° percentile del Laplaciano (leggermente filtrato dal rumore)."""
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.8), cv2.CV_32F))
    vals = lap[mask > 0] if mask is not None else lap.ravel()
    return float(np.percentile(vals, 90)) if vals.size else 0.0


def match_sharpness_and_noise(v_rgb, v_alpha_float, v_mask, context):
    """
    Porta la varroa alla stessa nitidezza e alla stessa grana della zona di
    foto in cui viene inserita (context): una varroa nitida su un'ape sfocata
    è il segnale più evidente di fotomontaggio.

    Colore e alpha vengono sfocati insieme, così anche il bordo si ammorbidisce
    quanto il resto. Restituisce (v_rgb, v_alpha_float).
    """
    if context.size == 0:
        return v_rgb, v_alpha_float

    ctx_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_sharpness = _sharpness(ctx_gray)

    # Misura la nitidezza della varroa solo al suo interno, lontano dal bordo
    inner = cv2.erode(v_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(inner) == 0:
        inner = v_mask
    size = np.sqrt(cv2.countNonZero(v_mask))
    max_sigma = max(MIN_BLUR_SIGMA, size * MAX_BLUR_RATIO)

    # Sfocatura minima che porta la varroa alla nitidezza del contesto
    sigma = MIN_BLUR_SIGMA
    while True:
        blurred = cv2.GaussianBlur(v_rgb, (0, 0), sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if _sharpness(gray, inner) <= target_sharpness or sigma >= max_sigma:
            break
        sigma = min(sigma + 0.25, max_sigma)
    alpha = cv2.GaussianBlur(v_alpha_float, (0, 0), sigma)

    # Grana: stima robusta (MAD) del rumore del contesto, tolte le basse frequenze
    residual = ctx_gray - cv2.GaussianBlur(ctx_gray, (0, 0), 1.5)
    noise_sigma = min(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), MAX_NOISE_SIGMA)
    noise = np.random.normal(0, noise_sigma, blurred.shape[:2] + (1,)).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy, alpha


def feather_alpha(alpha, radius=5):
    """
    Applica un feathering (sfumatura) naturale sul bordo della maschera alpha.
    """
    if radius <= 0:
        return alpha

    # Riduciamo molto di più il nucleo solido (in base al raggio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_core = cv2.erode(alpha, kernel, iterations=max(1, radius - 1))
    
    # Aumentiamo enormemente la dimensione del blur per fare un gradiente
    # che parta da dentro la varroa e finisca lentamente all'esterno
    blur_size = radius * 4 + 1
    blurred_alpha = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)
    
    # Per non renderla fantasma, il nucleo centrale rimane perfettamente solido (255)
    result = np.where(inner_core > 0, 255.0, blurred_alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_composite(target, src_rgb, alpha_float, x, y):
    """
    Compositing con alpha blending. Alpha è un float array [0,1].
    Il risultato è opaco dove alpha=1 (corpo varroa) e trasparente solo sui bordi sfumati.
    """
    h_s, w_s = src_rgb.shape[:2]
    h_t, w_t = target.shape[:2]

    # Clipping ai bordi dell'immagine target
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_s, w_t)
    y2 = min(y + h_s, h_t)

    if x1 >= x2 or y1 >= y2:
        return target

    # Offset nella sorgente
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    alpha = alpha_float[sy1:sy2, sx1:sx2, np.newaxis]  # (h, w, 1)
    rgb = src_rgb[sy1:sy2, sx1:sx2].astype(np.float32)

    roi = target[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + rgb * alpha
    target[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return target


def add_subtle_shadow(target, varroa_mask, insert_x, insert_y, offset=(2, 3), blur_size=7, opacity=0.3):
    """
    Aggiunge un'ombra sottile sotto la varroa per dare profondità.
    OTTIMIZZATA: calcola il blur solo sul rettangolo dell'ombra,
    evitando di sfocare milioni di pixel inutili.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]
    dx, dy = offset

    # Calcoliamo lo spazio extra per non tagliare l'ombra (blur radius + offset)
    pad = blur_size * 2 + max(dx, dy)
    
    # Creiamo un piccolo canvas temporaneo grande quanto la varroa + padding
    mini_canvas_h = rv_h + pad * 2
    mini_canvas_w = rv_w + pad * 2
    mini_canvas = np.zeros((mini_canvas_h, mini_canvas_w), dtype=np.uint8)
    
    # Incolliamo la maschera della varroa al centro del canvas temporaneo
    mini_canvas[pad:pad+rv_h, pad:pad+rv_w] = varroa_mask

    # Spostiamo l'ombra (offset) sul canvas temporaneo
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mini_canvas.astype(np.float32), M_shift, (mini_canvas_w, mini_canvas_h))

    # Sfocatura veloce SOLO sulla piccola porzione
    shadow_mask = cv2.GaussianBlur(shifted, (blur_size, blur_size), 0)
    shadow_mask = (shadow_mask / 255.0) * opacity

    # Calcoliamo dove incollare l'ombra sull'immagine globale
    sy1 = insert_y - pad
    sy2 = insert_y - pad + mini_canvas_h
    sx1 = insert_x - pad
    sx2 = insert_x - pad + mini_canvas_w

    # Clamp alle dimensioni dell'immagine target
    tgt_y1 = max(0, sy1)
    tgt_y2 = min(H, sy2)
    tgt_x1 = max(0, sx1)
    tgt_x2 = min(W, sx2)

    if tgt_y2 <= tgt_y1 or tgt_x2 <= tgt_x1:
        return target

    # Mappiamo le coordinate sul canvas temporaneo dell'ombra
    src_y1 = tgt_y1 - sy1
    src_y2 = src_y1 + (tgt_y2 - tgt_y1)
    src_x1 = tgt_x1 - sx1
    src_x2 = src_x1 + (tgt_x2 - tgt_x1)

    shadow_roi = shadow_mask[src_y1:src_y2, src_x1:src_x2]

    # Applica l'ombra scurendo il target locale
    for c in range(3):
        target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c] = np.clip(
            target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c].astype(np.float32) * (1.0 - shadow_roi), 0, 255
        ).astype(np.uint8)

    return target


def add_contact_shadow(target, varroa_mask, insert_x, insert_y, ring_width=3, opacity=0.18):
    """
    Aggiunge un alone scuro molto sottile attorno al bordo della varroa,
    simulando il contatto fisico del parassita con il corpo dell'ape.
    Molto leggero per non creare un bordo visibile.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]

    # Dilata la maschera per creare l'anello esterno
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    dilated = cv2.dilate(varroa_mask, kernel, iterations=1)

    # L'anello è la differenza tra dilatata e originale
    ring = dilated.astype(np.float32) - varroa_mask.astype(np.float32)

    # Blur più ampio per sfumare gradualmente nel corpo dell'ape
    blur_k = max(ring_width * 4 + 1, 5)
    if blur_k % 2 == 0:
        blur_k += 1
    ring = cv2.GaussianBlur(ring, (blur_k, blur_k), 0)
    ring = ring / 255.0 * opacity

    # Proietta sulla dimensione piena dell'immagine
    sy1 = max(insert_y, 0)
    sy2 = min(insert_y + rv_h, H)
    sx1 = max(insert_x, 0)
    sx2 = min(insert_x + rv_w, W)
    vy1 = sy1 - insert_y
    vy2 = vy1 + (sy2 - sy1)
    vx1 = sx1 - insert_x
    vx2 = vx1 + (sx2 - sx1)

    ring_roi = ring[vy1:vy2, vx1:vx2]

    # Scurisci il target nella zona dell'anello di contatto
    for c in range(3):
        target[sy1:sy2, sx1:sx2, c] = np.clip(
            target[sy1:sy2, sx1:sx2, c].astype(np.float32) * (1.0 - ring_roi),
            0, 255
        ).astype(np.uint8)

    return target


def overlaps_existing(gt_mask, v_mask_binary, x, y):
    """
    True se la varroa (maschera binaria) posizionata in (x, y) si sovrappone
    a una varroa già presente nella gt_mask.
    """
    H, W = gt_mask.shape[:2]
    rv_h, rv_w = v_mask_binary.shape[:2]
    y1, y2 = max(y, 0), min(y + rv_h, H)
    x1, x2 = max(x, 0), min(x + rv_w, W)
    if y2 <= y1 or x2 <= x1:
        return False
    new_varroa_region = v_mask_binary[y1 - y:y2 - y, x1 - x:x2 - x]
    return cv2.countNonZero(cv2.bitwise_and(gt_mask[y1:y2, x1:x2], new_varroa_region)) > 0


def find_body_position(favo, det, rv_w, rv_h, gt_mask=None, v_mask_binary=None, num_candidates=30):
    """
    Trova la posizione migliore sul corpo dell'ape (non sulle ali).
    Campiona più posizioni candidate dentro il poligono e sceglie quella
    con la saturazione più alta (il corpo dell'ape è più colorato/scuro,
    le ali sono chiare e poco sature).
    Scarta i candidati che si sovrappongono a varroa già inserite.

    rv_w, rv_h e v_mask_binary si riferiscono alla varroa ritagliata sulla sua
    parte opaca (senza padding). Restituisce (None, None) se non c'è posto.
    """
    bx, by, bw, bh = det["bbox"]
    polygon = det["polygon"].astype(np.int32)
    H, W = favo.shape[:2]
    check_overlap = gt_mask is not None and v_mask_binary is not None

    # Margini ridotti (25%) per lasciare spazio a più varroa sulla stessa ape
    margin_w = int(bw * 0.25)
    margin_h = int(bh * 0.25)

    x_min = bx + margin_w
    y_min = by + margin_h
    x_max = bx + bw - margin_w - rv_w
    y_max = by + bh - margin_h - rv_h

    if x_max <= x_min or y_max <= y_min:
        # Fallback: usa il centroide
        cx, cy = det["centroid"]
        fx, fy = int(cx - rv_w / 2), int(cy - rv_h / 2)
        if check_overlap and overlaps_existing(gt_mask, v_mask_binary, fx, fy):
            return None, None
        return fx, fy

    # OTTIMIZZAZIONE: Ritaglia l'area in cui si trova l'ape e converti solo quella in HSV.
    # Convertire 100x100 pixel è istantaneo rispetto ai 12 Megapixel dell'intera immagine.
    roi_y1, roi_y2 = max(0, by), min(H, by + bh)
    roi_x1, roi_x2 = max(0, bx), min(W, bx + bw)
    roi_bgr = favo[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi_bgr.size == 0:
        return None, None
        
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    best_score = -1
    best_x, best_y = x_min, y_min

    for _ in range(num_candidates):
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)
        
        # Il centro della varroa deve essere dentro il poligono
        center = (cx + rv_w // 2, cy + rv_h // 2)
        if cv2.pointPolygonTest(polygon, center, False) < 0:
            continue

        # Analizza la regione sotto la varroa
        ry1 = max(cy, 0)
        ry2 = min(cy + rv_h, H)
        rx1 = max(cx, 0)
        rx2 = min(cx + rv_w, W)

        if ry2 <= ry1 or rx2 <= rx1:
            continue

        # Check overlap
        if check_overlap and overlaps_existing(gt_mask, v_mask_binary, cx, cy):
            continue # Sovrapposizione, scarta candidato

        # Trasforma le coordinate globali (cx, cy) in coordinate locali alla ROI
        local_rx1 = rx1 - roi_x1
        local_rx2 = rx2 - roi_x1
        local_ry1 = ry1 - roi_y1
        local_ry2 = ry2 - roi_y1

        # Leggi la patch in formato HSV ma solo dalla ROI appena creata
        local_hsv = roi_hsv[local_ry1:local_ry2, local_rx1:local_rx2]
        
        if local_hsv.size == 0:
            continue
            
        sat_mean = local_hsv[:, :, 1].mean()
        val_mean = local_hsv[:, :, 2].mean()
        
        # Score = saturazione media + (255 - luminosità media)
        score = sat_mean + (255 - val_mean) * 0.5

        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy

    if best_score == -1:
        return None, None

    return best_x, best_y



def generate_synthetic_anomaly(favo, varroa_sources, detections):
    """
    Pipeline principale per l'inserimento realistico della varroa sulle api.
    Usa alpha compositing diretto (no Poisson blending) per evitare artefatti rettangolari.

    varroa_sources: lista di immagini RGBA delle varroa (normalizzate alla stessa scala).
    Per ogni inserimento viene scelta casualmente una delle varroa disponibili.
    """
    H, W = favo.shape[:2]
    synthetic_img = favo.copy()
    gt_mask = np.zeros((H, W), dtype=np.uint8)

    # Scegli quante api infettare
    num_bees = random.randint(NUM_BEES_TO_INFECT_MIN, NUM_BEES_TO_INFECT_MAX)

    # --- FILTRO API CANDIDATE: solo api ben visibili, isolate e non ai bordi ---
    BORDER_MARGIN = 40  # pixel dal bordo immagine
    MIN_BEE_SIDE = 60   # lato minimo bbox (più grande = ape più visibile)
    MAX_OVERLAP_RATIO = 0.15  # massima sovrapposizione con altre api (15%)

    valid_indices = []
    bboxes = [det["bbox"] for det in detections]

    for i, det in enumerate(detections):
        bx, by, bw, bh = det["bbox"]

        # 1. Scarta api troppo piccole
        if bw < MIN_BEE_SIDE or bh < MIN_BEE_SIDE:
            continue

        # 2. Scarta api il cui bounding box tocca i bordi dell'immagine
        if bx < BORDER_MARGIN or by < BORDER_MARGIN:
            continue
        if (bx + bw) > (W - BORDER_MARGIN) or (by + bh) > (H - BORDER_MARGIN):
            continue

        # 3. OTTIMIZZAZIONE: Usa la matematica dei Bounding Box per l'intersezione
        # invece di matrici (immagini) da 12 Megapixel. Calcolo fulmineo.
        overlaps_with_others = False
        this_area = bw * bh
        for j, other_bbox in enumerate(bboxes):
            if i == j: continue
            
            ox, oy, ow, oh = other_bbox
            
            # Calcola l'intersezione dei rettangoli
            ix = max(bx, ox)
            iy = max(by, oy)
            iw = min(bx + bw, ox + ow) - ix
            ih = min(by + bh, oy + oh) - iy
            
            if iw > 0 and ih > 0:
                # Se l'area sovrapposta supera la soglia, scarta l'ape
                overlap_ratio = (iw * ih) / this_area
                if overlap_ratio > MAX_OVERLAP_RATIO:
                    overlaps_with_others = True
                    break

        if overlaps_with_others:
            continue

        valid_indices.append(i)

    if not valid_indices:
        return None, None

    selected_indices = random.sample(valid_indices, min(num_bees, len(valid_indices)))

    print(f"    Infezione di {len(selected_indices)} api su {len(valid_indices)} valide (filtrate da {len(detections)} totali)...")
    inserted_count = 0
    for idx in selected_indices:
        det = detections[idx]

        # Scegli quante varroa mettere su questa ape
        varroa_count = random.randint(VARROA_PER_BEE_MIN, VARROA_PER_BEE_MAX)
        print(f"      Ape {idx}: tento di inserire {varroa_count} varroa...")

        for _v in range(varroa_count):
            bx, by, bw, bh = det["bbox"]

            # Scegli casualmente una delle varroa disponibili
            varroa_src = random.choice(varroa_sources)

            # --- 1. Dimensionamento della Varroa ---
            reference_dim = min(bw, bh)
            ratio = random.uniform(VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX)
            target_v_w = int(reference_dim * ratio)

            # Ridimensiona la varroa mantenendo l'aspect ratio: target_v_w è la
            # larghezza del corpo (le sorgenti sono normalizzate a VARROA_REF_BODY_WIDTH)
            varroa_resized = resize_rgba(varroa_src, target_v_w / VARROA_REF_BODY_WIDTH)

            # --- 2. Rotazione casuale con espansione ---
            angle = random.uniform(0, 360)
            varroa_rotated = rotate_image_rgba(varroa_resized, angle)

            # Crea maschera binaria netta per la GT mask e ritaglia la varroa sulla
            # sua parte opaca: posizionamento, sfumature e ombre vanno dimensionati
            # sulla varroa reale, non sul margine trasparente
            _, rot_mask = cv2.threshold(varroa_rotated[:, :, 3], 128, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(rot_mask) == 0:
                continue
            rot_mask = largest_component(rot_mask)
            tx, ty, rv_w, rv_h = cv2.boundingRect(rot_mask)
            v_mask_tight = rot_mask[ty:ty + rv_h, tx:tx + rv_w]
            size_ref = min(rv_w, rv_h)

            # Parametri di sfumatura e ombre proporzionali alla grandezza della varroa
            dynamic_feather = max(1, int(size_ref * 0.08))
            contact_ring_w = max(2, int(size_ref * 0.10))
            shadow_offset = max(1, int(size_ref * 0.08))
            shadow_blur = max(3, int(size_ref * 0.20)) | 1

            # --- FIX "EFFETTO RIQUADRO" ---
            # Aggiungiamo attorno alla varroa il margine trasparente strettamente
            # necessario a sfumatura e alone di contatto. In questo modo le sfocature
            # sfumano morbidamente a zero e non si "schiantano" contro il bordo dell'immagine.
            max_blur = max(MIN_BLUR_SIGMA, size_ref * MAX_BLUR_RATIO)
            pad_v = max(dynamic_feather * 2 + int(3 * max_blur) + 1, contact_ring_w * 3) + 2
            varroa_padded = cv2.copyMakeBorder(
                varroa_rotated[ty:ty + rv_h, tx:tx + rv_w], pad_v, pad_v, pad_v, pad_v,
                cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
            )
            v_mask_binary = cv2.copyMakeBorder(
                v_mask_tight, pad_v, pad_v, pad_v, pad_v, cv2.BORDER_CONSTANT, value=0
            )
            v_rgb = np.ascontiguousarray(varroa_padded[:, :, :3])

            # Crea alpha sfumato proporzionale alla grandezza della varroa
            v_alpha_feathered = feather_alpha(v_mask_binary, radius=dynamic_feather)
            v_alpha_float = v_alpha_feathered.astype(np.float32) / 255.0

            # --- 3. Posizionamento intelligente sul corpo dell'ape ---
            # (pos_x, pos_y) è l'angolo della varroa ritagliata, già verificato
            # contro le varroa presenti nella gt_mask
            pos_x, pos_y = find_body_position(
                favo, det, rv_w, rv_h, gt_mask=gt_mask, v_mask_binary=v_mask_tight, num_candidates=60
            )

            # Se non si è trovata una posizione senza overlap, salta questa varroa
            if pos_x is None or pos_y is None:
                continue

            # Angolo della patch con padding
            insert_x, insert_y = pos_x - pad_v, pos_y - pad_v

            # --- 4. Adattamento a colore, luce, nitidezza e grana dell'ape ---
            roi_y1 = max(pos_y, 0)
            roi_y2 = min(pos_y + rv_h, H)
            roi_x1 = max(pos_x, 0)
            roi_x2 = min(pos_x + rv_w, W)
            if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
                continue

            target_region = synthetic_img[roi_y1:roi_y2, roi_x1:roi_x2]

            # Sfondo allineato alla patch con padding (None se esce dall'immagine)
            ph, pw = v_mask_binary.shape[:2]
            bg_patch = None
            if insert_x >= 0 and insert_y >= 0 and insert_x + pw <= W and insert_y + ph <= H:
                bg_patch = synthetic_img[insert_y:insert_y + ph, insert_x:insert_x + pw]

            v_rgb_adapted = adapt_color_to_region(v_rgb, target_region, v_mask_binary, bg_patch=bg_patch)
            v_rgb_adapted = bleed_colors(v_rgb_adapted, v_mask_binary)

            # Zona di foto attorno alla varroa, da cui stimare nitidezza e grana
            context = synthetic_img[
                max(pos_y - rv_h, 0):min(pos_y + 2 * rv_h, H),
                max(pos_x - rv_w, 0):min(pos_x + 2 * rv_w, W)
            ]
            v_rgb_adapted, v_alpha_float = match_sharpness_and_noise(
                v_rgb_adapted, v_alpha_float, v_mask_binary, context
            )

            # Le ombre vanno applicate prima del compositing, altrimenti
            # scuriscono anche la varroa stessa

            # --- 5. Ombra di contatto ---
            synthetic_img = add_contact_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                ring_width=contact_ring_w, opacity=CONTACT_SHADOW_OPACITY
            )

            # --- 5b. Ombra sottile spostata ---
            synthetic_img = add_subtle_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                offset=(shadow_offset, shadow_offset),
                blur_size=shadow_blur,
                opacity=DROP_SHADOW_OPACITY
            )

            # --- 6. Alpha compositing diretto ---
            synthetic_img = alpha_composite(synthetic_img, v_rgb_adapted, v_alpha_float, insert_x, insert_y)

            # --- 7. Aggiorna la Ground Truth Mask ---
            gt_mask[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.bitwise_or(
                gt_mask[roi_y1:roi_y2, roi_x1:roi_x2],
                v_mask_tight[roi_y1 - pos_y:roi_y2 - pos_y, roi_x1 - pos_x:roi_x2 - pos_x]
            )

            inserted_count += 1

    if inserted_count == 0:
        return None, None

    _, gt_mask_final = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)

    return synthetic_img, gt_mask_final


# --- Main ---
if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    image_files = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(IMAGES_DIR, "*.[pP][nN][gG]"))
    )
    print(f"Trovate {len(image_files)} immagini in {IMAGES_DIR}")

    # Crea sottocartelle per immagini e maschere
    OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
    OUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    varroa_list = load_varroa_sources()
    print(f"{len(varroa_list)} varianti varroa caricate")

    for path_favo in image_files:
        base_name = os.path.splitext(os.path.basename(path_favo))[0]
        path_labels = os.path.join(LABELS_DIR, base_name + ".txt")

        if not os.path.exists(path_labels):
            print(f"Skipping {base_name}: label non trovata.")
            continue

        print(f"\nElaborazione di: {base_name}")
        favo_img = cv2.imread(path_favo)
        if favo_img is None:
            raise FileNotFoundError(f"Impossibile caricare l'immagine del favo: {path_favo}")
        H, W = favo_img.shape[:2]

        print(f"  Immagine caricata: {W}x{H}")

        # 1. Leggi le detection YOLO segmentation (poligoni)
        detections = parse_yolo_segmentation_labels(path_labels, W, H)

        print(f"  Rilevate {len(detections)} api.")

        if not detections:
            print(f"Nessuna ape rilevata in {base_name}. Skipping.")
            continue

        for i in range(NUM_VARIANTS):
            print(f"  Generazione variante {i}...")
            syn_img, gt_mask = generate_synthetic_anomaly(favo_img, varroa_list, detections)
            print(f"  Generazione completata per variante {i}.")

            if syn_img is not None:
                # Con più varianti l'indice va nel nome, altrimenti si sovrascrivono
                suffix = f"_{i}" if NUM_VARIANTS > 1 else ""
                out_img_path = os.path.join(OUT_IMG_DIR, f"{base_name}_syn{suffix}.jpg")
                out_mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}_mask{suffix}.png")
                cv2.imwrite(out_img_path, syn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(out_mask_path, gt_mask * 255)
                print(f"  ✓ Salvata: {out_img_path}")
            else:
                print(f"  ✗ Fallito inserimento su {base_name}.")

    print(f"\nPipeline completata. Controlla la cartella: {OUTPUT_DIR}")