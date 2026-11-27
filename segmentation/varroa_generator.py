from bee_paths import DATA_ROOT
import cv2
import numpy as np
import random
import os
import glob

# --- Configuration ---
IMAGES_DIR = DATA_ROOT + "/processed/test_batch3+4/images"
LABELS_DIR = DATA_ROOT + "/processed/test_batch3+4/labels"
VARROA_PATHS = [
    DATA_ROOT + "/processed/varroa_alpha.png",
    DATA_ROOT + "/processed/varroa_alpha2.png",
]
OUTPUT_DIR = DATA_ROOT + "/processed/test_batch3+4/synthetic_varroa_output"

# Seed to make the generation reproducible (None = random at every run)
SEED = 42

# How many synthetic variants to generate for each image
NUM_VARIANTS = 1

# Varroa/bee size ratio: Varroa destructor is ~1.6mm long,
# the bee's thorax is ~4-5mm, but in the photo the whole body takes up much more.
# We use the shorter side of the polygon's bounding box as the reference for the thorax.
VARROA_BEE_RATIO_MIN = 0.20
VARROA_BEE_RATIO_MAX = 0.24

# Width (in pixels) to which the body of every source varroa is normalised
VARROA_REF_BODY_WIDTH = 128

# How many bees to infest per image
NUM_BEES_TO_INFECT_MIN = 20
NUM_BEES_TO_INFECT_MAX = 40

# How many varroa to insert on each selected bee (1 to 2)
VARROA_PER_BEE_MIN = 1
VARROA_PER_BEE_MAX = 2

# --- Blending the varroa with the bee (0 = original varroa, 1 = full adaptation) ---
BLEND_LUMINANCE = 0.60   # how much the mean lightness follows that of the bee
BLEND_CONTRAST = 0.45    # how much the contrast follows that of the bee
BLEND_CHROMA = 0.45      # how much the colour shifts towards that of the bee
BLEND_SHADING = 0.60     # how much the varroa inherits the local highlights and shadows of the bee
# Blur (sigma in pixels) to match the sharpness of the photo:
# the minimum is always applied, the maximum is proportional to the varroa size
MIN_BLUR_SIGMA = 0.6
MAX_BLUR_RATIO = 0.07
# Cap on the added grain (std. dev.), estimated from the photo
MAX_NOISE_SIGMA = 6.0
# Shadow opacity
CONTACT_SHADOW_OPACITY = 0.25
DROP_SHADOW_OPACITY = 0.20


def resize_rgba(image_rgba, scale):
    """
    Resizes an RGBA image working in premultiplied alpha, so the
    (black) colour of the transparent pixels does not bleed into the varroa edges.
    """
    h, w = image_rgba.shape[:2]
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    # Use INTER_AREA for downscaling, INTER_CUBIC for upscaling
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
    Cleans up the mask of a source varroa: removes the light fringe
    left on the edge by the cut-out, removes thin hairs and legs (which
    would inflate the bounding box and pollute the GT mask) and closes the holes.
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
    """Loads all the RGBA varroa images.

    Each varroa is cleaned up, cropped to the body and normalised to the
    width VARROA_REF_BODY_WIDTH: this way differences in resolution and in
    transparent margin between the PNGs do not affect the final sizing
    on the bees.
    """
    normalized = []
    for vp in VARROA_PATHS:
        v = cv2.imread(vp, cv2.IMREAD_UNCHANGED)
        if v is None:
            raise FileNotFoundError(f"Could not load the varroa image: {vp}")
        if v.ndim != 3 or v.shape[2] != 4:
            raise ValueError(f"The varroa image must have an Alpha channel (32-bit PNG): {vp}")

        alpha = _clean_varroa_alpha(v[:, :, 3])
        if cv2.countNonZero(alpha) == 0:
            raise ValueError(f"The varroa image has no opaque pixels: {vp}")

        # Crop to the body
        x, y, w, h = cv2.boundingRect(alpha)
        orig_alpha = v[y:y + h, x:x + w, 3]
        alpha = alpha[y:y + h, x:x + w]
        rgb = np.ascontiguousarray(v[y:y + h, x:x + w, :3])

        # The holes closed in the mask were transparent in the PNG: fill them
        # with the surrounding colours
        holes = cv2.bitwise_and(alpha, cv2.bitwise_not(cv2.threshold(orig_alpha, 128, 255, cv2.THRESH_BINARY)[1]))
        if cv2.countNonZero(holes) > 0:
            rgb = cv2.inpaint(rgb, holes, 3, cv2.INPAINT_TELEA)

        normalized.append(resize_rgba(np.dstack([rgb, alpha]), VARROA_REF_BODY_WIDTH / w))

    return normalized


def parse_yolo_segmentation_labels(label_path, img_w, img_h):
    """
    Reads YOLO segmentation labels: each line is
       class_id x1 y1 x2 y2 ... xN yN
    where the coordinates are normalised to [0,1].

    Returns a list of dictionaries with:
      - 'polygon': np.array of points (N,2) in pixels
      - 'bbox': (x, y, w, h) bounding box in pixels
      - 'centroid': (cx, cy) centre of the polygon in pixels
    """
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (6 values)
                continue

            # The first value is the class_id, the rest are x,y pairs
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue  # Odd number of coordinates, skip

            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                points.append([px, py])

            polygon = np.array(points, dtype=np.float32)

            # Bounding box from the polygon
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            # Centroid
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
    Rotates an RGBA image by an arbitrary angle, expanding the bounding box
    so the edges are not cut off. The pixels added by the expansion are transparent.
    """
    h, w = image_rgba.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Compute the new size needed to contain the rotated image
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust the transformation matrix for the new centre
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Premultiplied alpha: interpolation does not mix the black of the transparent
    # pixels into the varroa edges
    rotated = cv2.warpAffine(
        _premultiply(image_rgba), M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return _unpremultiply(rotated)


def largest_component(mask):
    """
    Keeps only the largest connected component of the mask: after resizing and
    rotation, thin legs and hairs can detach from the body and would end up
    in the GT mask as isolated fragments.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    biggest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def bleed_colors(rgb, mask):
    """
    Extends the varroa colours beyond the mask edge, so the
    alpha feathering does not mix in the black of the transparent pixels.
    """
    return cv2.inpaint(rgb, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)


def adapt_color_to_region(varroa_rgb, target_region, alpha_mask, bg_patch=None):
    """
    Adapts the colour, lightness and contrast of the varroa to the photo of the bee.
    1. Match the contrast (standard deviation) as well as the mean lightness
    2. Blend the chrominance to absorb the reflections of the environment
    3. Transfer the local illumination of the bee (lit / shaded areas)

    bg_patch: portion of the image aligned pixel by pixel with varroa_rgb
    (same size); if None, step 3 is skipped.
    Sharpness and grain are matched afterwards, in match_sharpness_and_noise.
    """
    if target_region.size == 0 or varroa_rgb.size == 0:
        return varroa_rgb

    # Mask of the opaque varroa pixels
    opaque = alpha_mask > 128

    if opaque.sum() < 10:
        return varroa_rgb

    # Convert to the perceptual LAB colour space
    varroa_lab = cv2.cvtColor(varroa_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Extract the lightness (L) channels
    v_l = varroa_lab[:, :, 0][opaque]
    t_l = target_lab[:, :, 0]

    v_mean_l, v_std_l = v_l.mean(), v_l.std() + 1e-5
    t_mean_l, t_std_l = t_l.mean(), t_l.std() + 1e-5

    # 1. Contrast and lightness matching
    target_contrast = (v_std_l * (1.0 - BLEND_CONTRAST) + t_std_l * BLEND_CONTRAST) / v_std_l
    l_shift = (t_mean_l - v_mean_l) * BLEND_LUMINANCE
    new_l = (varroa_lab[:, :, 0] - v_mean_l) * target_contrast + v_mean_l + l_shift

    # 3. Local illumination: the varroa follows the slow light variations
    # of the underlying bee instead of having a uniform lightness
    if bg_patch is not None and bg_patch.shape[:2] == varroa_rgb.shape[:2]:
        bg_l = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        sigma = max(2.0, np.sqrt(opaque.sum()) * 0.25)
        bg_low = cv2.GaussianBlur(bg_l, (0, 0), sigma)
        shading = bg_low / max(bg_low[opaque].mean(), 1.0)
        new_l = new_l * (1.0 + BLEND_SHADING * (shading - 1.0))

    # Apply the transformation to the L channel
    varroa_lab[:, :, 0] = np.where(opaque, np.clip(new_l, 0, 255), varroa_lab[:, :, 0])

    # 2. Chrominance matching
    for ch in [1, 2]:
        v_ch = varroa_lab[:, :, ch][opaque]
        t_ch = target_lab[:, :, ch]
        # Partial shift to preserve the typical colour of the parasite
        shift = (t_ch.mean() - v_ch.mean()) * BLEND_CHROMA
        varroa_lab[:, :, ch] = np.where(
            opaque,
            np.clip(varroa_lab[:, :, ch] + shift, 0, 255),
            varroa_lab[:, :, ch]
        )

    # Convert back to BGR
    return cv2.cvtColor(varroa_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _sharpness(gray, mask=None):
    """Sharpness measure: 90th percentile of the Laplacian (lightly filtered for noise)."""
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.8), cv2.CV_32F))
    vals = lap[mask > 0] if mask is not None else lap.ravel()
    return float(np.percentile(vals, 90)) if vals.size else 0.0


def match_sharpness_and_noise(v_rgb, v_alpha_float, v_mask, context):
    """
    Brings the varroa to the same sharpness and grain as the area of the
    photo where it is inserted (context): a sharp varroa on a blurry bee
    is the most obvious sign of a photomontage.

    Colour and alpha are blurred together, so the edge softens as much
    as the rest. Returns (v_rgb, v_alpha_float).
    """
    if context.size == 0:
        return v_rgb, v_alpha_float

    ctx_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_sharpness = _sharpness(ctx_gray)

    # Measure the varroa sharpness only inside it, away from the edge
    inner = cv2.erode(v_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(inner) == 0:
        inner = v_mask
    size = np.sqrt(cv2.countNonZero(v_mask))
    max_sigma = max(MIN_BLUR_SIGMA, size * MAX_BLUR_RATIO)

    # Minimum blur that brings the varroa down to the sharpness of the context
    sigma = MIN_BLUR_SIGMA
    while True:
        blurred = cv2.GaussianBlur(v_rgb, (0, 0), sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if _sharpness(gray, inner) <= target_sharpness or sigma >= max_sigma:
            break
        sigma = min(sigma + 0.25, max_sigma)
    alpha = cv2.GaussianBlur(v_alpha_float, (0, 0), sigma)

    # Grain: robust (MAD) estimate of the context noise, with low frequencies removed
    residual = ctx_gray - cv2.GaussianBlur(ctx_gray, (0, 0), 1.5)
    noise_sigma = min(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), MAX_NOISE_SIGMA)
    noise = np.random.normal(0, noise_sigma, blurred.shape[:2] + (1,)).astype(np.float32)
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy, alpha


def feather_alpha(alpha, radius=5):
    """
    Applies natural feathering to the edge of the alpha mask.
    """
    if radius <= 0:
        return alpha

    # Shrink the solid core much more (based on the radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_core = cv2.erode(alpha, kernel, iterations=max(1, radius - 1))
    
    # Greatly increase the blur size to create a gradient
    # that starts inside the varroa and fades out slowly outside
    blur_size = radius * 4 + 1
    blurred_alpha = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)
    
    # So it does not look ghostly, the central core stays perfectly solid (255)
    result = np.where(inner_core > 0, 255.0, blurred_alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_composite(target, src_rgb, alpha_float, x, y):
    """
    Compositing with alpha blending. Alpha is a float array in [0,1].
    The result is opaque where alpha=1 (varroa body) and transparent only on the feathered edges.
    """
    h_s, w_s = src_rgb.shape[:2]
    h_t, w_t = target.shape[:2]

    # Clip to the borders of the target image
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_s, w_t)
    y2 = min(y + h_s, h_t)

    if x1 >= x2 or y1 >= y2:
        return target

    # Offset in the source
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
    Adds a subtle shadow under the varroa to give depth.
    OPTIMISED: computes the blur only on the shadow rectangle,
    avoiding blurring millions of useless pixels.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]
    dx, dy = offset

    # Compute the extra space so the shadow is not cut off (blur radius + offset)
    pad = blur_size * 2 + max(dx, dy)
    
    # Create a small temporary canvas as large as the varroa + padding
    mini_canvas_h = rv_h + pad * 2
    mini_canvas_w = rv_w + pad * 2
    mini_canvas = np.zeros((mini_canvas_h, mini_canvas_w), dtype=np.uint8)
    
    # Paste the varroa mask in the centre of the temporary canvas
    mini_canvas[pad:pad+rv_h, pad:pad+rv_w] = varroa_mask

    # Shift the shadow (offset) on the temporary canvas
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mini_canvas.astype(np.float32), M_shift, (mini_canvas_w, mini_canvas_h))

    # Fast blur ONLY on the small portion
    shadow_mask = cv2.GaussianBlur(shifted, (blur_size, blur_size), 0)
    shadow_mask = (shadow_mask / 255.0) * opacity

    # Compute where to paste the shadow on the full image
    sy1 = insert_y - pad
    sy2 = insert_y - pad + mini_canvas_h
    sx1 = insert_x - pad
    sx2 = insert_x - pad + mini_canvas_w

    # Clamp to the size of the target image
    tgt_y1 = max(0, sy1)
    tgt_y2 = min(H, sy2)
    tgt_x1 = max(0, sx1)
    tgt_x2 = min(W, sx2)

    if tgt_y2 <= tgt_y1 or tgt_x2 <= tgt_x1:
        return target

    # Map the coordinates onto the temporary shadow canvas
    src_y1 = tgt_y1 - sy1
    src_y2 = src_y1 + (tgt_y2 - tgt_y1)
    src_x1 = tgt_x1 - sx1
    src_x2 = src_x1 + (tgt_x2 - tgt_x1)

    shadow_roi = shadow_mask[src_y1:src_y2, src_x1:src_x2]

    # Apply the shadow by darkening the local target
    for c in range(3):
        target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c] = np.clip(
            target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c].astype(np.float32) * (1.0 - shadow_roi), 0, 255
        ).astype(np.uint8)

    return target


def add_contact_shadow(target, varroa_mask, insert_x, insert_y, ring_width=3, opacity=0.18):
    """
    Adds a very thin dark halo around the edge of the varroa,
    simulating the physical contact of the parasite with the bee's body.
    Very light, so as not to create a visible border.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]

    # Dilate the mask to create the outer ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    dilated = cv2.dilate(varroa_mask, kernel, iterations=1)

    # The ring is the difference between the dilated and the original mask
    ring = dilated.astype(np.float32) - varroa_mask.astype(np.float32)

    # Wider blur to fade gradually into the bee's body
    blur_k = max(ring_width * 4 + 1, 5)
    if blur_k % 2 == 0:
        blur_k += 1
    ring = cv2.GaussianBlur(ring, (blur_k, blur_k), 0)
    ring = ring / 255.0 * opacity

    # Project onto the full image size
    sy1 = max(insert_y, 0)
    sy2 = min(insert_y + rv_h, H)
    sx1 = max(insert_x, 0)
    sx2 = min(insert_x + rv_w, W)
    vy1 = sy1 - insert_y
    vy2 = vy1 + (sy2 - sy1)
    vx1 = sx1 - insert_x
    vx2 = vx1 + (sx2 - sx1)

    ring_roi = ring[vy1:vy2, vx1:vx2]

    # Darken the target in the contact-ring area
    for c in range(3):
        target[sy1:sy2, sx1:sx2, c] = np.clip(
            target[sy1:sy2, sx1:sx2, c].astype(np.float32) * (1.0 - ring_roi),
            0, 255
        ).astype(np.uint8)

    return target


def overlaps_existing(gt_mask, v_mask_binary, x, y):
    """
    True if the varroa (binary mask) placed at (x, y) overlaps
    a varroa already present in gt_mask.
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
    Finds the best position on the bee's body (not on the wings).
    Samples several candidate positions inside the polygon and picks the one
    with the highest saturation (the bee's body is more colourful/darker,
    the wings are light and low-saturation).
    Discards candidates that overlap varroa already inserted.

    rv_w, rv_h and v_mask_binary refer to the varroa cropped to its
    opaque part (without padding). Returns (None, None) if there is no room.
    """
    bx, by, bw, bh = det["bbox"]
    polygon = det["polygon"].astype(np.int32)
    H, W = favo.shape[:2]
    check_overlap = gt_mask is not None and v_mask_binary is not None

    # Reduced margins (25%) to leave room for several varroa on the same bee
    margin_w = int(bw * 0.25)
    margin_h = int(bh * 0.25)

    x_min = bx + margin_w
    y_min = by + margin_h
    x_max = bx + bw - margin_w - rv_w
    y_max = by + bh - margin_h - rv_h

    if x_max <= x_min or y_max <= y_min:
        # Fallback: use the centroid
        cx, cy = det["centroid"]
        fx, fy = int(cx - rv_w / 2), int(cy - rv_h / 2)
        if check_overlap and overlaps_existing(gt_mask, v_mask_binary, fx, fy):
            return None, None
        return fx, fy

    # OPTIMISATION: crop the area where the bee is and convert only that to HSV.
    # Converting 100x100 pixels is instantaneous compared with the 12 megapixels of the whole image.
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
        
        # The centre of the varroa must be inside the polygon
        center = (cx + rv_w // 2, cy + rv_h // 2)
        if cv2.pointPolygonTest(polygon, center, False) < 0:
            continue

        # Analyse the region under the varroa
        ry1 = max(cy, 0)
        ry2 = min(cy + rv_h, H)
        rx1 = max(cx, 0)
        rx2 = min(cx + rv_w, W)

        if ry2 <= ry1 or rx2 <= rx1:
            continue

        # Check overlap
        if check_overlap and overlaps_existing(gt_mask, v_mask_binary, cx, cy):
            continue # Overlap, discard the candidate

        # Transform the global coordinates (cx, cy) into coordinates local to the ROI
        local_rx1 = rx1 - roi_x1
        local_rx2 = rx2 - roi_x1
        local_ry1 = ry1 - roi_y1
        local_ry2 = ry2 - roi_y1

        # Read the HSV patch, but only from the ROI just created
        local_hsv = roi_hsv[local_ry1:local_ry2, local_rx1:local_rx2]
        
        if local_hsv.size == 0:
            continue
            
        sat_mean = local_hsv[:, :, 1].mean()
        val_mean = local_hsv[:, :, 2].mean()
        
        # Score = mean saturation + (255 - mean brightness)
        score = sat_mean + (255 - val_mean) * 0.5

        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy

    if best_score == -1:
        return None, None

    return best_x, best_y



def generate_synthetic_anomaly(favo, varroa_sources, detections):
    """
    Main pipeline for the realistic insertion of varroa on the bees.
    Uses direct alpha compositing (no Poisson blending) to avoid rectangular artefacts.

    varroa_sources: list of RGBA varroa images (normalised to the same scale).
    For each insertion one of the available varroa is chosen at random.
    """
    H, W = favo.shape[:2]
    synthetic_img = favo.copy()
    gt_mask = np.zeros((H, W), dtype=np.uint8)

    # Choose how many bees to infest
    num_bees = random.randint(NUM_BEES_TO_INFECT_MIN, NUM_BEES_TO_INFECT_MAX)

    # --- CANDIDATE BEE FILTER: only clearly visible, isolated bees away from the borders ---
    BORDER_MARGIN = 40  # pixels from the image border
    MIN_BEE_SIDE = 60   # minimum bbox side (larger = more visible bee)
    MAX_OVERLAP_RATIO = 0.15  # maximum overlap with other bees (15%)

    valid_indices = []
    bboxes = [det["bbox"] for det in detections]

    for i, det in enumerate(detections):
        bx, by, bw, bh = det["bbox"]

        # 1. Discard bees that are too small
        if bw < MIN_BEE_SIDE or bh < MIN_BEE_SIDE:
            continue

        # 2. Discard bees whose bounding box touches the image borders
        if bx < BORDER_MARGIN or by < BORDER_MARGIN:
            continue
        if (bx + bw) > (W - BORDER_MARGIN) or (by + bh) > (H - BORDER_MARGIN):
            continue

        # 3. OPTIMISATION: use bounding-box arithmetic for the intersection
        # instead of 12-megapixel matrices (images). Lightning fast.
        overlaps_with_others = False
        this_area = bw * bh
        for j, other_bbox in enumerate(bboxes):
            if i == j: continue
            
            ox, oy, ow, oh = other_bbox
            
            # Compute the intersection of the rectangles
            ix = max(bx, ox)
            iy = max(by, oy)
            iw = min(bx + bw, ox + ow) - ix
            ih = min(by + bh, oy + oh) - iy
            
            if iw > 0 and ih > 0:
                # If the overlapping area exceeds the threshold, discard the bee
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

    print(f"    Infesting {len(selected_indices)} bees out of {len(valid_indices)} valid ones (filtered from {len(detections)} in total)...")
    inserted_count = 0
    for idx in selected_indices:
        det = detections[idx]

        # Choose how many varroa to put on this bee
        varroa_count = random.randint(VARROA_PER_BEE_MIN, VARROA_PER_BEE_MAX)
        print(f"      Bee {idx}: trying to insert {varroa_count} varroa...")

        for _v in range(varroa_count):
            bx, by, bw, bh = det["bbox"]

            # Randomly choose one of the available varroa
            varroa_src = random.choice(varroa_sources)

            # --- 1. Varroa sizing ---
            reference_dim = min(bw, bh)
            ratio = random.uniform(VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX)
            target_v_w = int(reference_dim * ratio)

            # Resize the varroa keeping the aspect ratio: target_v_w is the
            # body width (the sources are normalised to VARROA_REF_BODY_WIDTH)
            varroa_resized = resize_rgba(varroa_src, target_v_w / VARROA_REF_BODY_WIDTH)

            # --- 2. Random rotation with expansion ---
            angle = random.uniform(0, 360)
            varroa_rotated = rotate_image_rgba(varroa_resized, angle)

            # Create a sharp binary mask for the GT mask and crop the varroa to its
            # opaque part: placement, feathering and shadows must be sized
            # on the real varroa, not on the transparent margin
            _, rot_mask = cv2.threshold(varroa_rotated[:, :, 3], 128, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(rot_mask) == 0:
                continue
            rot_mask = largest_component(rot_mask)
            tx, ty, rv_w, rv_h = cv2.boundingRect(rot_mask)
            v_mask_tight = rot_mask[ty:ty + rv_h, tx:tx + rv_w]
            size_ref = min(rv_w, rv_h)

            # Feathering and shadow parameters proportional to the varroa size
            dynamic_feather = max(1, int(size_ref * 0.08))
            contact_ring_w = max(2, int(size_ref * 0.10))
            shadow_offset = max(1, int(size_ref * 0.08))
            shadow_blur = max(3, int(size_ref * 0.20)) | 1

            # --- FIX FOR THE "BOX EFFECT" ---
            # Add around the varroa the transparent margin strictly
            # needed for feathering and the contact halo. This way the blurs
            # fade smoothly to zero and do not "crash" into the image border.
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

            # Create a feathered alpha proportional to the varroa size
            v_alpha_feathered = feather_alpha(v_mask_binary, radius=dynamic_feather)
            v_alpha_float = v_alpha_feathered.astype(np.float32) / 255.0

            # --- 3. Smart placement on the bee's body ---
            # (pos_x, pos_y) is the corner of the cropped varroa, already checked
            # against the varroa present in gt_mask
            pos_x, pos_y = find_body_position(
                favo, det, rv_w, rv_h, gt_mask=gt_mask, v_mask_binary=v_mask_tight, num_candidates=60
            )

            # If no position without overlap was found, skip this varroa
            if pos_x is None or pos_y is None:
                continue

            # Corner of the padded patch
            insert_x, insert_y = pos_x - pad_v, pos_y - pad_v

            # --- 4. Adaptation to the colour, lighting, sharpness and grain of the bee ---
            roi_y1 = max(pos_y, 0)
            roi_y2 = min(pos_y + rv_h, H)
            roi_x1 = max(pos_x, 0)
            roi_x2 = min(pos_x + rv_w, W)
            if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
                continue

            target_region = synthetic_img[roi_y1:roi_y2, roi_x1:roi_x2]

            # Background aligned with the padded patch (None if it goes outside the image)
            ph, pw = v_mask_binary.shape[:2]
            bg_patch = None
            if insert_x >= 0 and insert_y >= 0 and insert_x + pw <= W and insert_y + ph <= H:
                bg_patch = synthetic_img[insert_y:insert_y + ph, insert_x:insert_x + pw]

            v_rgb_adapted = adapt_color_to_region(v_rgb, target_region, v_mask_binary, bg_patch=bg_patch)
            v_rgb_adapted = bleed_colors(v_rgb_adapted, v_mask_binary)

            # Area of the photo around the varroa, from which sharpness and grain are estimated
            context = synthetic_img[
                max(pos_y - rv_h, 0):min(pos_y + 2 * rv_h, H),
                max(pos_x - rv_w, 0):min(pos_x + 2 * rv_w, W)
            ]
            v_rgb_adapted, v_alpha_float = match_sharpness_and_noise(
                v_rgb_adapted, v_alpha_float, v_mask_binary, context
            )

            # The shadows must be applied before compositing, otherwise
            # they also darken the varroa itself

            # --- 5. Contact shadow ---
            synthetic_img = add_contact_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                ring_width=contact_ring_w, opacity=CONTACT_SHADOW_OPACITY
            )

            # --- 5b. Subtle drop shadow ---
            synthetic_img = add_subtle_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                offset=(shadow_offset, shadow_offset),
                blur_size=shadow_blur,
                opacity=DROP_SHADOW_OPACITY
            )

            # --- 6. Direct alpha compositing ---
            synthetic_img = alpha_composite(synthetic_img, v_rgb_adapted, v_alpha_float, insert_x, insert_y)

            # --- 7. Update the ground-truth mask ---
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
    print(f"Found {len(image_files)} images in {IMAGES_DIR}")

    # Create subfolders for images and masks
    OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
    OUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    varroa_list = load_varroa_sources()
    print(f"{len(varroa_list)} varroa variants loaded")

    for path_favo in image_files:
        base_name = os.path.splitext(os.path.basename(path_favo))[0]
        path_labels = os.path.join(LABELS_DIR, base_name + ".txt")

        if not os.path.exists(path_labels):
            print(f"Skipping {base_name}: label not found.")
            continue

        print(f"\nProcessing: {base_name}")
        favo_img = cv2.imread(path_favo)
        if favo_img is None:
            raise FileNotFoundError(f"Could not load the comb image: {path_favo}")
        H, W = favo_img.shape[:2]

        print(f"  Image loaded: {W}x{H}")

        # 1. Read the YOLO segmentation detections (polygons)
        detections = parse_yolo_segmentation_labels(path_labels, W, H)

        print(f"  Detected {len(detections)} bees.")

        if not detections:
            print(f"No bees detected in {base_name}. Skipping.")
            continue

        for i in range(NUM_VARIANTS):
            print(f"  Generating variant {i}...")
            syn_img, gt_mask = generate_synthetic_anomaly(favo_img, varroa_list, detections)
            print(f"  Generation complete for variant {i}.")

            if syn_img is not None:
                # With several variants the index goes into the name, otherwise they overwrite each other
                suffix = f"_{i}" if NUM_VARIANTS > 1 else ""
                out_img_path = os.path.join(OUT_IMG_DIR, f"{base_name}_syn{suffix}.jpg")
                out_mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}_mask{suffix}.png")
                cv2.imwrite(out_img_path, syn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(out_mask_path, gt_mask * 255)
                print(f"  ✓ Saved: {out_img_path}")
            else:
                print(f"  ✗ Insertion failed on {base_name}.")

    print(f"\nPipeline complete. Check the folder: {OUTPUT_DIR}")