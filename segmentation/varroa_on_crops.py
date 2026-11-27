"""
Prova: inserimento della varroa direttamente sulle singole api già segmentate
(crop 512x512 a sfondo nero), invece che sull'immagine del favo prima della
segmentazione.

Riusa le funzioni di varroa_generator.py (caricamento sorgenti, adattamento
colore, ombre, compositing) senza modificarlo. Cambia solo ciò che dipende dal
fatto di lavorare su un crop:
  - la maschera dell'ape si ricava dai pixel non neri del crop;
  - la varroa viene posizionata solo sul corpo dell'ape, fino a subito sotto
    gli occhi (mai su ali, occhi, zampe o sfondo);
  - colore, nitidezza e grana sono stimati solo sui pixel dell'ape, così lo
    sfondo nero e il bordo dell'ape non falsano le statistiche.

Output: per ogni ape il crop con varroa, la GT mask e un foglio di confronto.
"""

from bee_paths import DATA_ROOT
import argparse
import glob
import os
import random

import cv2
import numpy as np

import varroa_generator as vg

# --- Configurazione ---
CROPS_DIR = DATA_ROOT + "/processed/train_single_bee_batch3+4_clusters/Prova1/Crop_Utilizzabile"
OUTPUT_DIR = DATA_ROOT + "/processed/prova_varroa_su_crop"

SEED = 42
NUM_BEES = 12

# Larghezza della varroa rispetto alla larghezza dell'ape (lato corto del
# rettangolo orientato della maschera)
VARROA_BEE_RATIO_MIN = 0.20
VARROA_BEE_RATIO_MAX = 0.24

VARROA_PER_BEE_MIN = 1
VARROA_PER_BEE_MAX = 2

# Tentativi (dimensione e rotazione nuove) per ogni varroa quando il numero è imposto
ATTEMPTS_PER_VARROA = 5

# Soglia sotto cui un pixel del crop è considerato sfondo
BG_THRESHOLD = 10

# --- Zona ammessa: corpo dell'ape, esclusi ali, zampe e occhi ---
# Il corpo è la parte "spessa" della maschera: resta dopo un'apertura con raggio
# pari a questa frazione dello spessore massimo (ali sottili, zampe e antenne spariscono)
BODY_CORE_RATIO = 0.60
# Frazione della lunghezza del corpo esclusa a ciascuna estremità: la testa può
# essere in alto o in basso, quindi si tolgono entrambe (occhi e punta dell'addome).
# La zona subito sotto gli occhi resta ammessa.
BODY_END_MARGIN = 0.22
# Distanza massima dall'asse del corpo, in frazione della semilarghezza del corpo
BODY_AXIS_BAND = 0.70
# Pixel chiari e poco saturi = ala (membrana traslucida e riflessi)
WING_MAX_SATURATION = 75
WING_MIN_VALUE = 125
# Frazione della varroa che deve cadere nella zona ammessa (1.0 = tutta)
MIN_ZONE_COVERAGE = 1.0

# I crop sono api piccole ingrandite a 512: sono molto più morbidi del favo
# originale, quindi alla varroa serve una sfocatura massima più ampia
MAX_BLUR_RATIO = 0.12


def bee_mask_from_crop(crop):
    """Maschera dell'ape: componente più grande dei pixel non neri, senza buchi."""
    mask = (crop.max(axis=2) > BG_THRESHOLD).astype(np.uint8) * 255
    mask = vg.largest_component(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def bee_width(mask):
    """Larghezza dell'ape: lato corto del rettangolo orientato della maschera."""
    pts = cv2.findNonZero(mask)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return min(w, h)


def body_zone(crop, bee_mask):
    """
    Zona in cui la varroa può stare: il corpo dell'ape (torace e addome) fino a
    subito sotto gli occhi. Sono esclusi ali, zampe, antenne e le due estremità
    dell'ape (testa con gli occhi e punta dell'addome).
    """
    zone = np.zeros_like(bee_mask)
    dist = cv2.distanceTransform(bee_mask, cv2.DIST_L2, 5)
    radius = int(dist.max() * BODY_CORE_RATIO)
    if radius < 2:
        return zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    core = vg.largest_component(cv2.morphologyEx(bee_mask, cv2.MORPH_OPEN, kernel))
    if cv2.countNonZero(core) == 0:
        return zone

    # Asse del corpo: direzione principale della parte spessa
    ys, xs = np.nonzero(core)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    center = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - center).T))
    axis, normal = vecs[:, 1], vecs[:, 0]
    along = (pts - center) @ axis
    across = (pts - center) @ normal
    lo, hi = along.min(), along.max()
    margin = (hi - lo) * BODY_END_MARGIN
    half_width = np.percentile(np.abs(across), 95)
    keep = (along >= lo + margin) & (along <= hi - margin) & (np.abs(across) <= half_width * BODY_AXIS_BAND)
    zone[ys[keep], xs[keep]] = 255

    # Via le ali: membrana chiara e poco satura
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (0, 0), 3), cv2.COLOR_BGR2HSV)
    wing = (hsv[:, :, 1] < WING_MAX_SATURATION) & (hsv[:, :, 2] > WING_MIN_VALUE)
    wing = cv2.morphologyEx(wing.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    zone[wing > 0] = 0
    return zone


def find_position_in_mask(stats_img, bee_mask, zone, gt_mask, v_mask_tight, num_candidates=200):
    """
    Sceglie l'angolo in alto a sinistra della varroa in modo che stia nella
    zona ammessa (corpo dell'ape, vedi body_zone). Tra i candidati vince quello
    sulla zona più satura/scura, come in varroa_generator.
    """
    rv_h, rv_w = v_mask_tight.shape[:2]
    H, W = bee_mask.shape[:2]
    v_area = cv2.countNonZero(v_mask_tight)
    ys, xs = np.nonzero(zone)
    if len(ys) == 0:
        return None, None

    hsv = cv2.cvtColor(stats_img, cv2.COLOR_BGR2HSV)
    best_score, best = -1, (None, None)
    for i in np.random.choice(len(ys), size=min(num_candidates, len(ys)), replace=False):
        x, y = int(xs[i] - rv_w // 2), int(ys[i] - rv_h // 2)
        if x < 0 or y < 0 or x + rv_w > W or y + rv_h > H:
            continue
        if vg.overlaps_existing(gt_mask, v_mask_tight, x, y):
            continue
        covered = cv2.countNonZero(cv2.bitwise_and(zone[y:y + rv_h, x:x + rv_w], v_mask_tight))
        if covered < MIN_ZONE_COVERAGE * v_area:
            continue
        patch = hsv[y:y + rv_h, x:x + rv_w]
        score = patch[:, :, 1].mean() + (255 - patch[:, :, 2].mean()) * 0.5
        if score > best_score:
            best_score, best = score, (x, y)
    return best


def match_sharpness_and_noise(v_rgb, v_alpha_float, v_mask, context, context_mask, max_blur_ratio):
    """
    Come vg.match_sharpness_and_noise, ma nitidezza e grana del contesto sono
    misurate solo sui pixel dell'ape (context_mask).
    """
    inside = cv2.erode(context_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if cv2.countNonZero(inside) < 50:
        return v_rgb, v_alpha_float

    ctx_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_sharpness = vg._sharpness(ctx_gray, inside)

    inner = cv2.erode(v_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(inner) == 0:
        inner = v_mask
    max_sigma = max(vg.MIN_BLUR_SIGMA, np.sqrt(cv2.countNonZero(v_mask)) * max_blur_ratio)

    sigma = vg.MIN_BLUR_SIGMA
    while True:
        blurred = cv2.GaussianBlur(v_rgb, (0, 0), sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if vg._sharpness(gray, inner) <= target_sharpness or sigma >= max_sigma:
            break
        sigma = min(sigma + 0.25, max_sigma)
    alpha = cv2.GaussianBlur(v_alpha_float, (0, 0), sigma)

    residual = (ctx_gray - cv2.GaussianBlur(ctx_gray, (0, 0), 1.5))[inside > 0]
    noise_sigma = min(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), vg.MAX_NOISE_SIGMA)
    # Il crop è ingrandito, quindi la sua grana è "larga": si sfoca anche il
    # rumore aggiunto e lo si riporta alla deviazione standard stimata
    noise = cv2.GaussianBlur(np.random.normal(0, 1, blurred.shape[:2]).astype(np.float32), (0, 0), sigma)
    noise = (noise / (noise.std() + 1e-6) * noise_sigma)[:, :, np.newaxis]
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, alpha


def add_varroa_to_crop(crop, varroa_sources, ratio_range, max_blur_ratio=MAX_BLUR_RATIO, match_blur=True,
                       num_varroa=None):
    """
    Inserisce le varroa su un singolo crop, solo nella zona ammessa (body_zone).
    Restituisce (crop_sintetico, gt_mask, zona) con le maschere a 0/255;
    crop e gt_mask sono None se non c'è posto.

    num_varroa: None = 1-2 varroa a caso, tenendo quelle che entrano. Un numero
    = esattamente quel numero di varroa (con più tentativi per ciascuna),
    altrimenti l'inserimento fallisce.
    """
    H, W = crop.shape[:2]
    bee_mask = bee_mask_from_crop(crop)
    if cv2.countNonZero(bee_mask) == 0:
        return None, None, None

    zone = body_zone(crop, bee_mask)

    out = crop.copy()
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    width = bee_width(bee_mask)

    if num_varroa is None:
        wanted = random.randint(VARROA_PER_BEE_MIN, VARROA_PER_BEE_MAX)
        attempts = wanted
    else:
        wanted = num_varroa
        attempts = num_varroa * ATTEMPTS_PER_VARROA

    inserted = 0
    for _ in range(attempts):
        if inserted == wanted:
            break
        # --- 1. Dimensione e rotazione ---
        target_v_w = int(width * random.uniform(*ratio_range))
        varroa = vg.resize_rgba(random.choice(varroa_sources), target_v_w / vg.VARROA_REF_BODY_WIDTH)
        varroa = vg.rotate_image_rgba(varroa, random.uniform(0, 360))

        _, rot_mask = cv2.threshold(varroa[:, :, 3], 128, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(rot_mask) == 0:
            continue
        rot_mask = vg.largest_component(rot_mask)
        tx, ty, rv_w, rv_h = cv2.boundingRect(rot_mask)
        v_mask_tight = rot_mask[ty:ty + rv_h, tx:tx + rv_w]
        size_ref = min(rv_w, rv_h)

        feather = max(1, int(size_ref * 0.08))
        contact_ring_w = max(2, int(size_ref * 0.10))
        shadow_offset = max(1, int(size_ref * 0.08))
        shadow_blur = max(3, int(size_ref * 0.20)) | 1

        max_blur = max(vg.MIN_BLUR_SIGMA, size_ref * max_blur_ratio)
        pad = max(feather * 2 + int(3 * max_blur) + 1, contact_ring_w * 3) + 2
        varroa_padded = cv2.copyMakeBorder(
            varroa[ty:ty + rv_h, tx:tx + rv_w], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
        )
        v_mask = cv2.copyMakeBorder(v_mask_tight, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        v_rgb = np.ascontiguousarray(varroa_padded[:, :, :3])
        v_alpha = vg.feather_alpha(v_mask, radius=feather).astype(np.float32) / 255.0

        # Immagine per le statistiche: lo sfondo nero è sostituito dal colore
        # medio dell'ape, così medie e sfocature non vengono trascinate verso il nero
        stats_img = out.copy()
        stats_img[bee_mask == 0] = np.array(cv2.mean(out, mask=bee_mask)[:3], dtype=np.uint8)
        stats_img = cv2.copyMakeBorder(stats_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # --- 2. Posizione dentro l'ape ---
        pos_x, pos_y = find_position_in_mask(stats_img[pad:-pad, pad:-pad], bee_mask, zone, gt_mask, v_mask_tight)
        if pos_x is None:
            continue
        insert_x, insert_y = pos_x - pad, pos_y - pad

        # --- 3. Colore, luce, nitidezza e grana dell'ape ---
        ph, pw = v_mask.shape[:2]
        target_region = stats_img[pos_y + pad:pos_y + pad + rv_h, pos_x + pad:pos_x + pad + rv_w]
        bg_patch = stats_img[pos_y:pos_y + ph, pos_x:pos_x + pw]
        v_rgb = vg.adapt_color_to_region(v_rgb, target_region, v_mask, bg_patch=bg_patch)
        v_rgb = vg.bleed_colors(v_rgb, v_mask)

        if match_blur:
            cy1, cy2 = max(pos_y - rv_h, 0), min(pos_y + 2 * rv_h, H)
            cx1, cx2 = max(pos_x - rv_w, 0), min(pos_x + 2 * rv_w, W)
            v_rgb, v_alpha = match_sharpness_and_noise(
                v_rgb, v_alpha, v_mask, out[cy1:cy2, cx1:cx2], bee_mask[cy1:cy2, cx1:cx2], max_blur_ratio
            )

        # --- 4. Ombre e compositing ---
        out = vg.add_contact_shadow(out, v_mask, insert_x, insert_y,
                                    ring_width=contact_ring_w, opacity=vg.CONTACT_SHADOW_OPACITY)
        out = vg.add_subtle_shadow(out, v_mask, insert_x, insert_y, offset=(shadow_offset, shadow_offset),
                                   blur_size=shadow_blur, opacity=vg.DROP_SHADOW_OPACITY)
        out = vg.alpha_composite(out, v_rgb, v_alpha, insert_x, insert_y)

        gt_mask[pos_y:pos_y + rv_h, pos_x:pos_x + rv_w] = cv2.bitwise_or(
            gt_mask[pos_y:pos_y + rv_h, pos_x:pos_x + rv_w], v_mask_tight
        )
        inserted += 1

    if inserted == 0 or (num_varroa is not None and inserted != num_varroa):
        return None, None, zone
    return out, gt_mask, zone


def zoom_on_mask(img, mask, size):
    """Ritaglio quadrato attorno alle varroa, ingrandito a size x size."""
    x, y, w, h = cv2.boundingRect(mask)
    side = int(max(w, h, 90) * 1.5)
    cx, cy = x + w // 2, y + h // 2
    H, W = img.shape[:2]
    x1 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
    y1 = int(np.clip(cy - side // 2, 0, max(H - side, 0)))
    return cv2.resize(img[y1:y1 + side, x1:x1 + side], (size, size), interpolation=cv2.INTER_CUBIC)


def comparison_row(original, synthetic, gt_mask, zone, tile=384):
    """Riga di confronto: originale | con varroa | zoom | zona ammessa (azzurro) e GT mask (verde)."""
    overlay = synthetic.copy()
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 200, 0), 2)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    tiles = [
        cv2.resize(original, (tile, tile), interpolation=cv2.INTER_AREA),
        cv2.resize(synthetic, (tile, tile), interpolation=cv2.INTER_AREA),
        zoom_on_mask(synthetic, gt_mask, tile),
        cv2.resize(overlay, (tile, tile), interpolation=cv2.INTER_AREA),
    ]
    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="base", help="nome della prova (sottocartella di output)")
    parser.add_argument("--num-bees", type=int, default=NUM_BEES)
    parser.add_argument("--ratio", type=float, nargs=2, default=[VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX])
    parser.add_argument("--max-blur-ratio", type=float, default=MAX_BLUR_RATIO)
    parser.add_argument("--no-match-blur", action="store_true", help="non adattare nitidezza e grana")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(CROPS_DIR, "*.png")))
    # Stesse api per ogni prova con lo stesso seed, così le prove sono confrontabili
    selected = random.Random(args.seed).sample(files, args.num_bees)

    out_dir = os.path.join(OUTPUT_DIR, args.name)
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    varroa_sources = vg.load_varroa_sources()
    rows = []
    for path in selected:
        crop = cv2.imread(path)
        base = os.path.splitext(os.path.basename(path))[0]
        syn, gt, zone = add_varroa_to_crop(crop, varroa_sources, args.ratio,
                                     max_blur_ratio=args.max_blur_ratio, match_blur=not args.no_match_blur)
        if syn is None:
            print(f"  ✗ {base}: nessuna posizione valida")
            continue
        cv2.imwrite(os.path.join(out_dir, "images", f"{base}_syn.png"), syn)
        cv2.imwrite(os.path.join(out_dir, "masks", f"{base}_mask.png"), gt)
        rows.append(comparison_row(crop, syn, gt, zone))
        print(f"  ✓ {base}: {cv2.connectedComponents(gt)[0] - 1} varroa")

    if rows:
        sheet_path = os.path.join(out_dir, "confronto.jpg")
        cv2.imwrite(sheet_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nFoglio di confronto: {sheet_path}")


if __name__ == "__main__":
    main()
