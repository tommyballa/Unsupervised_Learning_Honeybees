"""
Trial: inserting the varroa directly onto the single, already segmented bees
(512x512 crops on a black background), instead of onto the comb image before
segmentation.

Reuses the functions of varroa_generator.py (source loading, colour
adaptation, shadows, compositing) without modifying it. Only what depends on
working on a crop changes:
  - the bee mask is derived from the non-black pixels of the crop;
  - the varroa is placed only on the bee's body, up to just below
    the eyes (never on wings, eyes, legs or background);
  - colour, sharpness and grain are estimated only on the bee pixels, so the
    black background and the bee's edge do not skew the statistics.

Output: for each bee the crop with varroa, the GT mask and a comparison sheet.
"""

from bee_paths import DATA_ROOT
import argparse
import glob
import os
import random

import cv2
import numpy as np

import varroa_generator as vg

# --- Configuration ---
CROPS_DIR = DATA_ROOT + "/processed/train_single_bee_batch3+4_clusters/Prova1/Crop_Utilizzabile"
OUTPUT_DIR = DATA_ROOT + "/processed/prova_varroa_su_crop"

SEED = 42
NUM_BEES = 12

# Varroa width relative to the bee width (short side of the
# oriented rectangle of the mask)
VARROA_BEE_RATIO_MIN = 0.20
VARROA_BEE_RATIO_MAX = 0.24

VARROA_PER_BEE_MIN = 1
VARROA_PER_BEE_MAX = 2

# Attempts (with new size and rotation) for each varroa when the number is fixed
ATTEMPTS_PER_VARROA = 5

# Threshold below which a crop pixel is considered background
BG_THRESHOLD = 10

# --- Allowed zone: the bee's body, excluding wings, legs and eyes ---
# The body is the "thick" part of the mask: it is what remains after an opening with radius
# equal to this fraction of the maximum thickness (thin wings, legs and antennae disappear)
BODY_CORE_RATIO = 0.60
# Fraction of the body length excluded at each end: the head can
# be at the top or at the bottom, so both are removed (eyes and tip of the abdomen).
# The area just below the eyes stays allowed.
BODY_END_MARGIN = 0.22
# Maximum distance from the body axis, as a fraction of the body half-width
BODY_AXIS_BAND = 0.70
# Light, low-saturation pixels = wing (translucent membrane and reflections)
WING_MAX_SATURATION = 75
WING_MIN_VALUE = 125
# Fraction of the varroa that must fall inside the allowed zone (1.0 = all of it)
MIN_ZONE_COVERAGE = 1.0

# The crops are small bees upscaled to 512: they are much softer than the original
# comb, so the varroa needs a larger maximum blur
MAX_BLUR_RATIO = 0.12


def bee_mask_from_crop(crop):
    """Bee mask: largest component of the non-black pixels, without holes."""
    mask = (crop.max(axis=2) > BG_THRESHOLD).astype(np.uint8) * 255
    mask = vg.largest_component(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def bee_width(mask):
    """Bee width: short side of the oriented rectangle of the mask."""
    pts = cv2.findNonZero(mask)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return min(w, h)


def body_zone(crop, bee_mask):
    """
    Zone where the varroa may sit: the bee's body (thorax and abdomen) up to
    just below the eyes. Wings, legs, antennae and the two ends of the bee
    (head with the eyes and tip of the abdomen) are excluded.
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

    # Body axis: principal direction of the thick part
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

    # Remove the wings: light, low-saturation membrane
    hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (0, 0), 3), cv2.COLOR_BGR2HSV)
    wing = (hsv[:, :, 1] < WING_MAX_SATURATION) & (hsv[:, :, 2] > WING_MIN_VALUE)
    wing = cv2.morphologyEx(wing.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    zone[wing > 0] = 0
    return zone


def find_position_in_mask(stats_img, bee_mask, zone, gt_mask, v_mask_tight, num_candidates=200):
    """
    Chooses the top-left corner of the varroa so that it lies in the
    allowed zone (the bee's body, see body_zone). Among the candidates, the one
    on the most saturated/darkest area wins, as in varroa_generator.
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
    Like vg.match_sharpness_and_noise, but the sharpness and grain of the context are
    measured only on the bee pixels (context_mask).
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
    # The crop is upscaled, so its grain is "coarse": the added noise is blurred
    # too and brought back to the estimated standard deviation
    noise = cv2.GaussianBlur(np.random.normal(0, 1, blurred.shape[:2]).astype(np.float32), (0, 0), sigma)
    noise = (noise / (noise.std() + 1e-6) * noise_sigma)[:, :, np.newaxis]
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, alpha


def add_varroa_to_crop(crop, varroa_sources, ratio_range, max_blur_ratio=MAX_BLUR_RATIO, match_blur=True,
                       num_varroa=None):
    """
    Inserts the varroa on a single crop, only in the allowed zone (body_zone).
    Returns (synthetic_crop, gt_mask, zone) with the masks at 0/255;
    crop and gt_mask are None if there is no room.

    num_varroa: None = 1-2 varroa at random, keeping those that fit. A number
    = exactly that number of varroa (with several attempts for each one),
    otherwise the insertion fails.
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
        # --- 1. Size and rotation ---
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

        # Image for the statistics: the black background is replaced by the mean
        # colour of the bee, so means and blurs are not dragged towards black
        stats_img = out.copy()
        stats_img[bee_mask == 0] = np.array(cv2.mean(out, mask=bee_mask)[:3], dtype=np.uint8)
        stats_img = cv2.copyMakeBorder(stats_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # --- 2. Position inside the bee ---
        pos_x, pos_y = find_position_in_mask(stats_img[pad:-pad, pad:-pad], bee_mask, zone, gt_mask, v_mask_tight)
        if pos_x is None:
            continue
        insert_x, insert_y = pos_x - pad, pos_y - pad

        # --- 3. Colour, lighting, sharpness and grain of the bee ---
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

        # --- 4. Shadows and compositing ---
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
    """Square crop around the varroa, upscaled to size x size."""
    x, y, w, h = cv2.boundingRect(mask)
    side = int(max(w, h, 90) * 1.5)
    cx, cy = x + w // 2, y + h // 2
    H, W = img.shape[:2]
    x1 = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
    y1 = int(np.clip(cy - side // 2, 0, max(H - side, 0)))
    return cv2.resize(img[y1:y1 + side, x1:x1 + side], (size, size), interpolation=cv2.INTER_CUBIC)


def comparison_row(original, synthetic, gt_mask, zone, tile=384):
    """Comparison row: original | with varroa | zoom | allowed zone (light blue) and GT mask (green)."""
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
    parser.add_argument("--name", default="base", help="name of the trial (output subfolder)")
    parser.add_argument("--num-bees", type=int, default=NUM_BEES)
    parser.add_argument("--ratio", type=float, nargs=2, default=[VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX])
    parser.add_argument("--max-blur-ratio", type=float, default=MAX_BLUR_RATIO)
    parser.add_argument("--no-match-blur", action="store_true", help="do not match sharpness and grain")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(CROPS_DIR, "*.png")))
    # Same bees for every trial with the same seed, so the trials are comparable
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
            print(f"  ✗ {base}: no valid position")
            continue
        cv2.imwrite(os.path.join(out_dir, "images", f"{base}_syn.png"), syn)
        cv2.imwrite(os.path.join(out_dir, "masks", f"{base}_mask.png"), gt)
        rows.append(comparison_row(crop, syn, gt, zone))
        print(f"  ✓ {base}: {cv2.connectedComponents(gt)[0] - 1} varroa")

    if rows:
        sheet_path = os.path.join(out_dir, "confronto.jpg")
        cv2.imwrite(sheet_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nComparison sheet: {sheet_path}")


if __name__ == "__main__":
    main()
