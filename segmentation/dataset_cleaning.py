from bee_paths import DATA_ROOT
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
# CLEANING PARAMETERS (sanitisation)
# ============================================================
MIN_BLUR_SCORE = 50.0       # Below this threshold the image is too blurry (score at a common scale, see blur_score)
MIN_AREA_PIXELS = 1500      # Minimum mask area (in pixels) to be a real bee
MIN_FILL_RATIO = 0.30       # A bee fills at least 30% of its minimum rotated rectangle
MAX_ASPECT_RATIO = 3.0      # Maximum length/width ratio (avoids fragments such as wings/legs)
MIN_BBOX_SIDE = 25          # Minimum short side of the minimum rotated rectangle (in pixels of the original crop)
# Below ~100 px of length the crops are almost only bee pieces (abdomens in cells, heads,
# wings, covered bees) that pass the area and aspect ratio filters; between 100 and 120 px they are mixed.
MIN_BEE_LENGTH = 105        # Minimum bee length (long side of the rotated rectangle, pixels of the original crop)

# Bees arrive at very different scales (from ~130 to ~490 px long). The morphological
# kernels below are expressed for a bee REF_BEE_LENGTH px long and are
# rescaled in proportion to the length of each bee.
REF_BEE_LENGTH = 190        # Reference length (peak of the distribution of extracted bees)
KERNEL_SCALE_RANGE = (0.5, 3.0)
CLOSE_KERNEL = 15           # Closes the holes in the raw polygon
REFINE_DILATE_KERNEL = 11   # Dilation so hairs and legs are not cut off
OPEN_KERNEL = 25            # Opening that breaks the thin bridges between adjacent bees
RECOVER_DILATE_KERNEL = 19  # Dilation that recovers legs and antennae after the opening
SMOOTH_KERNEL = 31          # Blur+threshold that rounds the contours
MAX_FRAGMENT_LOSS = 0.20    # Maximum fraction of the mask lost by keeping only the main body
RAW_BORDER_MARGIN = 8       # Band (px) along the crop border in which to look for a cut-off bee, raw masks
LEGACY_BORDER_MARGIN = 3    # Same for legacy masks, already dilated during extraction
MAX_BORDER_PIXELS = 15      # Mask pixels tolerated in that band

# ============================================================
# NORMALISATION PARAMETERS
# ============================================================
TARGET_SIZE = 512           # Final size of the square image
TARGET_FILL_RATIO = 0.85    # The long side of the bee fills ~85% of the frame after normalisation
EDGE_FEATHER_PX = 12        # Pixels of gradual feathering at the mask edges
# Head at the top (bee_orientation.py). In run_cleaning it is a separate step, batched on
# GPU, on the bees already saved: it has its own record in the manifest and does not invalidate the cleaning.
ORIENT_HEAD_UP = True

# To be incremented when the pipeline logic changes (not just the parameters):
# together with the parameters it decides whether the manifest of a previous run is still valid.
PIPELINE_VERSION = 4

# Execution: they do not affect the output, so they do not invalidate the manifest
NUM_WORKERS = min(16, os.cpu_count() or 1)   # Parallel processes
MANIFEST_SAVE_EVERY = 500                    # How often (in files) to save the manifest

# Name of the manifest file that tracks the files already processed (accepted and rejected)
CLEANING_MANIFEST_FILENAME = "_cleaning_manifest.json"


class CleaningResult(NamedTuple):
    """Outcome of cleaning one bee. If rejected, image/mask/matrix are None and reason says why."""
    image: Optional[np.ndarray]    # TARGET_SIZE x TARGET_SIZE x 3, ready to be saved
    mask: Optional[np.ndarray]     # bee mask in the same frame as the final image
    matrix: Optional[np.ndarray]   # 2x3 affine original crop -> final frame (to align other masks)
    reason: Optional[str]          # None if accepted, otherwise the filter that rejected it


def cleaning_params():
    """Parameters that determine the output: if they change, the previous manifest is no longer valid."""
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
    Loads the cleaning manifest (files already processed: accepted and rejected).

    The manifest is only valid for the parameters it was written with: if they changed
    (or it comes from an earlier version that did not record them) it starts from scratch, so
    all files are reprocessed with the current parameters.
    """
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        if manifest.get("params") == cleaning_params():
            manifest.setdefault("discard_reasons", {})
            return manifest
        print("⚠️  Cleaning parameters changed compared with the existing manifest: reprocessing all files.")
    return new_cleaning_manifest()


def save_cleaning_manifest(output_dir, manifest):
    """Saves the updated cleaning manifest."""
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_image_and_mask(path):
    """
    Loads an image and its mask.

    Supports two formats:
    - RGBA PNG (new): the Alpha channel contains the raw mask of the YOLO polygon,
      preserved without any processing from the extraction. This avoids the loss
      of information caused by re-deriving the mask from the pixels.
    - RGB PNG/JPG (legacy): the mask is derived by thresholding the non-black pixels.
      Compatibility with images extracted by the previous version of the script.

    Returns:
        img_bgr (H x W x 3), mask (H x W), is_raw_polygon (bool)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, False

    if img.ndim == 3 and img.shape[2] == 4:
        # RGBA: take the mask from the alpha channel (raw data from the YOLO polygon)
        mask = img[:, :, 3]
        img_bgr = img[:, :, :3]
        return img_bgr, mask, True
    else:
        # Legacy RGB: derive the mask by thresholding (fallback for old extractions)
        img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return img_bgr, mask, False


def largest_contour(mask):
    """Largest external contour of the mask, or None if it is empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def bee_length(mask):
    """Bee length: long side of the minimum rotated rectangle (does not depend on orientation)."""
    contour = largest_contour(mask)
    if contour is None:
        return 0.0
    return max(cv2.minAreaRect(contour)[1])


def kernel_scale(length):
    """Factor by which to rescale the kernels for a bee of this length."""
    return float(np.clip(length / REF_BEE_LENGTH, *KERNEL_SCALE_RANGE))


def scaled_size(base, scale):
    """Odd size of a kernel designed for REF_BEE_LENGTH, rescaled."""
    return max(3, int(round(base * scale))) | 1


def scaled_ellipse(base, scale):
    size = scaled_size(base, scale)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def refine_raw_polygon_mask(mask, scale=1.0):
    """
    Refines the raw mask of the YOLO polygon.

    Applies the morphological operations that improve the continuity
    and coverage of the mask. These operations used to be
    performed both in extract_anomaly_dataset.py and here, causing
    double processing. Now they are centralised here only.

    Pipeline:
    1. Morphological closing → fills internal holes and fractures in the polygon
    2. Dilation → expands the mask to cover hairs and legs

    This function is called ONLY for raw masks (RGBA).
    Legacy (RGB) images already went through these operations
    during extraction and are NOT reprocessed.

    scale: kernel factor (see kernel_scale), 1.0 = bee REF_BEE_LENGTH long.
    """
    # Temporary black border: without it, a closing with a kernel larger than the crop
    # padding would leave the mask attached to the border (erosion does not act from outside).
    pad = scaled_size(CLOSE_KERNEL, scale)
    h, w = mask.shape[:2]
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    # 1. Morphological closing: fills internal holes and joins indentations (fractures)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, scaled_ellipse(CLOSE_KERNEL, scale))
    
    # 2. Expand the mask so hairs and legs are not cut off
    mask = cv2.dilate(mask, scaled_ellipse(REFINE_DILATE_KERNEL, scale), iterations=1)
    
    return mask[pad:pad + h, pad:pad + w]


def blur_score(img_bgr, mask, length):
    """
    Bee sharpness: variance of the Laplacian on the mask pixels only,
    measured at a common scale.

    At native resolution the score drops with the size of the bee (the same
    details are spread over more pixels) and a fixed threshold rejects large bees
    even when they are in focus. Bees longer than REF_BEE_LENGTH are therefore
    downscaled to that length before measuring; shorter ones stay at native
    resolution, because upscaling them would lower the score through
    interpolation alone.
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
    2x3 affine that brings the bee into the final frame: major axis vertical, long side
    equal to target_fill * target_size, centred in the target_size x target_size square.

    Rotation, scale and centring are in a single transformation, so the image
    is interpolated only once.

    Returns:
        (M, scale) or (None, None) if the mask is empty
    """
    contour = largest_contour(mask)
    if contour is None:
        return None, None

    # minAreaRect returns (centre, (width, height), angle)
    (cx, cy), (w_rect, h_rect), angle = cv2.minAreaRect(contour)
    long_side = max(w_rect, h_rect)
    if long_side <= 0:
        return None, None

    # Rotating by `angle` straightens the rectangle, with w_rect along x and h_rect along y.
    # If width > height the major axis is horizontal: another 90° is needed to make it vertical.
    rotation_angle = angle + 90 if w_rect > h_rect else angle
    scale = (target_size * target_fill) / long_side

    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, scale)
    # The centre of the rectangle (i.e. of the bee) goes to the centre of the final frame
    M[0, 2] += target_size / 2 - cx
    M[1, 2] += target_size / 2 - cy
    return M, scale


def feather_edges(img_bgr, mask, feather_px=12):
    """
    Gradually feathers the mask edges to create a smooth transition
    between the bee and the black background, removing the hard cut.

    The result is an image where the pixels at the edges of the bee "fade"
    gently towards black, reducing edge artefacts for PatchCore.

    Returns:
        img_feathered (same size)
    """
    # Create an "eroded" version of the mask (the core that is certainly inside the bee)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
    inner_mask = cv2.erode(mask, erode_kernel, iterations=1)

    # Create a smooth gradient:
    # - Inside (inner_mask) = 1.0 (fully visible)
    # - Edge = gradual fade from 1.0 to 0.0
    # - Outside = 0.0 (fully black)
    # To do so, blur the original mask with a large kernel
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (feather_px * 4 + 1, feather_px * 4 + 1), 0)

    # Force the inside to be full (1.0) so the bee itself is not blurred
    alpha[inner_mask > 0] = 1.0

    # Apply the alpha to the image
    alpha_3c = np.stack([alpha] * 3, axis=-1)
    img_feathered = (img_bgr.astype(np.float32) * alpha_3c).astype(np.uint8)

    return img_feathered


def _discard(reason):
    return CleaningResult(None, None, None, reason)


def rotate_frame_180(result):
    """
    Rotates an accepted bee by 180°: image, mask and affine stay consistent.

    cv2.rotate maps pixel (x, y) to (S-1-x, S-1-y); composing the same map with
    the affine keeps anyone who reapplies it (e.g. to varroa masks) aligned.
    """
    flip = np.array([[-1.0, 0.0, TARGET_SIZE - 1], [0.0, -1.0, TARGET_SIZE - 1], [0.0, 0.0, 1.0]])
    matrix = (flip @ np.vstack([result.matrix, [0.0, 0.0, 1.0]]))[:2]
    return CleaningResult(cv2.rotate(result.image, cv2.ROTATE_180), cv2.rotate(result.mask, cv2.ROTATE_180),
                          matrix, None)


def clean_bee(img_bgr, mask, is_raw_polygon, orienter=None):
    """
    Applies all the sanitisation filters and the normalisation to a single bee.

    Args:
        img_bgr: BGR image (original crop resolution)
        mask: binary mask (from the RGBA alpha channel or from the legacy threshold)
        is_raw_polygon: True if the mask is the raw YOLO polygon (RGBA),
                        False if it has already been pre-processed (legacy RGB)
        orienter: bee_orientation.HeadOrienter to put the head at the top; if None
                  the bee keeps its vertical axis but the head may be at the bottom
                  (run_cleaning orients it afterwards, in batches)

    Returns:
        CleaningResult: cleaned and normalised image, its mask and the affine
        used; or only the rejection reason.
    """

    # The morphological kernels follow the size of the bee (measured on the input mask)
    length = bee_length(mask)
    if length <= 0: return _discard("empty_mask")
    if length < MIN_BEE_LENGTH: return _discard("short")
    k = kernel_scale(length)

    # --- FILTER 1: Bee cut off at the borders ---
    # Must be done on the input mask: after refinement the dilation, which grows
    # with the size of the bee, would bring even whole bees close to the border. For raw
    # polygons the margin is wider, as when the check came after a 5 px dilation.
    h_img, w_img = mask.shape[:2]
    border_margin = RAW_BORDER_MARGIN if is_raw_polygon else LEGACY_BORDER_MARGIN
    border_pixels = (cv2.countNonZero(mask[:border_margin, :]) + cv2.countNonZero(mask[h_img - border_margin:, :]) +
                     cv2.countNonZero(mask[:, :border_margin]) + cv2.countNonZero(mask[:, w_img - border_margin:]))
    if border_pixels > MAX_BORDER_PIXELS: return _discard("border")
    
    # --- MASK REFINEMENT (raw RGBA polygons only) ---
    # For (new) RGBA images, apply closing + dilation to the raw polygon
    # mask. For (legacy) RGB images, these operations were already
    # done in extract_anomaly_dataset.py, so we skip them.
    if is_raw_polygon:
        mask = refine_raw_polygon_mask(mask, k)

    # --- FILTER 2: Break thin bridges ---
    # Large kernel to forcefully separate adjacent or connected bees
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, scaled_ellipse(OPEN_KERNEL, k))

    body_contour = largest_contour(opened_mask)
    if body_contour is None: return _discard("no_body")

    # --- FILTER 3: Fragment removal ---
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [body_contour], -1, 255, thickness=cv2.FILLED)

    # Dilation to recover legs and antennae after the opening
    clean_mask = cv2.dilate(clean_mask, scaled_ellipse(RECOVER_DILATE_KERNEL, k), iterations=1)
    clean_mask = cv2.bitwise_and(clean_mask, mask)

    total_mask_area = cv2.countNonZero(mask)
    main_area = cv2.countNonZero(clean_mask)
    if total_mask_area > 0:
        if (1.0 - (main_area / total_mask_area)) > MAX_FRAGMENT_LOSS: return _discard("fragments")

    # Close the internal holes for good by filling the external contours
    contours_clean, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_clean:
        cv2.drawContours(clean_mask, contours_clean, -1, 255, thickness=cv2.FILLED)

    # --- FILTER 4: Minimum area ---
    area = cv2.countNonZero(clean_mask)
    if area < MIN_AREA_PIXELS: return _discard("area")

    # --- FILTER 5: Aspect ratio ---
    # Measured on the minimum rotated rectangle: with the axis-aligned bounding box a diagonal
    # bee would have a ratio of ~1 and a low fill whatever its shape.
    rect_w, rect_h = cv2.minAreaRect(largest_contour(clean_mask))[1]
    long_side, short_side = max(rect_w, rect_h), min(rect_w, rect_h)
    if short_side < MIN_BBOX_SIDE: return _discard("side")
    if long_side / short_side > MAX_ASPECT_RATIO: return _discard("aspect")

    # --- FILTER 6: Fill ratio ---
    if area / (long_side * short_side) < MIN_FILL_RATIO: return _discard("fill")

    # --- FILTER 7: Blur score ---
    if blur_score(img_bgr, clean_mask, long_side) < MIN_BLUR_SCORE: return _discard("blur")

    # --- CONTOUR SMOOTHING ---
    # A single blur+threshold pass to round all the edges
    # (it used to be done both in extract and here — now only here, once)
    smooth_size = scaled_size(SMOOTH_KERNEL, k)
    clean_mask = cv2.GaussianBlur(clean_mask, (smooth_size, smooth_size), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)

    # ============================================================
    # NORMALISATION
    # ============================================================

    # 1. ORIENTATION + SCALE: major axis vertical, long side at ~85% of the 512x512 frame.
    #    This is the ONLY interpolation in the pipeline: rotation and resize are a single affine,
    #    applied to the original pixels of the crop.
    M, scale = normalization_matrix(clean_mask)
    if M is None: return _discard("no_body")

    out_size = (TARGET_SIZE, TARGET_SIZE)
    cleaned_img = cv2.warpAffine(img_bgr, M, out_size,
                                 flags=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    clean_mask = cv2.warpAffine(clean_mask, M, out_size, flags=cv2.INTER_NEAREST, borderValue=0)
    cleaned_img = cv2.bitwise_and(cleaned_img, cleaned_img, mask=clean_mask)

    # Brightness and contrast are not touched here: they were already equalised over the whole
    # photo in extract_anomaly_dataset.py, and a per-bee correction would undo that.

    # 2. EDGE FEATHERING: smooth transition bee → black background
    cleaned_img = feather_edges(cleaned_img, clean_mask, feather_px=EDGE_FEATHER_PX)

    result = CleaningResult(cleaned_img, clean_mask, M, None)

    # 3. HEAD AT THE TOP: exact 180° rotation if the classifier sees the head at the bottom
    if orienter is not None and orienter.scores([cleaned_img])[0] < 0:
        result = rotate_frame_180(result)
    return result


def clean_and_filter(img_bgr, mask, is_raw_polygon):
    """Like clean_bee, but returns only the final image (None if the bee is rejected)."""
    return clean_bee(img_bgr, mask, is_raw_polygon).image


def _init_worker():
    # The images are small: OpenCV's internal threads cost more than they give back
    # and would get in each other's way across processes.
    cv2.setNumThreads(1)


def _clean_file(job):
    """Cleans a single file (run in the worker processes). Returns (filename, status, reason, removed)."""
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
        # A bee accepted by a run with different parameters must not stay in the dataset
        os.remove(out_path)
        removed = True
    return filename, status, reason, removed


def orient_saved_bees(output_dir, manifest, device="auto", batch_size=256, workers=NUM_WORKERS):
    """
    Puts the head at the top for the bees accepted and already saved in output_dir.

    Runs in batches on GPU after the cleaning, which works on CPU in several processes. The step is
    idempotent (an already straightened bee has a positive score and is not touched), so
    an interruption halfway through cannot flip the same bee twice. The manifest keeps
    the score of each bee before straightening: negative = it was rotated,
    close to 0 = uncertain orientation.
    """
    from concurrent.futures import ThreadPoolExecutor
    from bee_orientation import HeadOrienter

    orienter = HeadOrienter(device=device)
    log = manifest.get("orientation")
    if not log or log.get("weights_id") != orienter.weights_id:
        # New weights: all bees are re-checked (the ones already correct stay as they are)
        log = {"weights_id": orienter.weights_id, "scores": {}}
    manifest["orientation"] = log

    todo = sorted(f for f, status in manifest["processed_files"].items()
                  if status == "accepted" and f not in log["scores"])
    print(f"Head-up orientation on {orienter.device}: {len(todo)} bees to check.")

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
                print(f"  Oriented {start + len(names)}/{len(todo)} bees...")

    save_cleaning_manifest(output_dir, manifest)
    print(f"Rotated {rotated} bees by 180° out of {len(todo)} checked.")


def run_cleaning(input_dir, output_dir, workers=NUM_WORKERS, device="auto"):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))

    # Ignore the extraction manifest file if it is in the input folder
    img_paths = [p for p in img_paths if not os.path.basename(p).startswith("_")]

    # Load the cleaning manifest to know which files have already been processed
    manifest = load_cleaning_manifest(output_dir)

    # Skip the files already processed (accepted or rejected) in a previous run
    jobs = [(p, os.path.join(output_dir, os.path.basename(p))) for p in img_paths
            if os.path.basename(p) not in manifest["processed_files"]]

    total = len(img_paths)
    skipped = total - len(jobs)
    discarded = 0
    saved = 0
    removed = 0
    print(f"Found {total} images in '{input_dir}' ({skipped} already processed).")
    print(f"Starting cleaning and normalisation with {workers} processes...")

    # The time is dominated by disk reads and writes: several processes in parallel
    # keep the disk busy instead of waiting for it one file at a time.
    # "spawn" and not "fork": a child process created by fork after the parent has used
    # OpenCV inherits its thread pool in an unusable state and hangs.
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

            # Save the manifest periodically, to be robust to interruptions
            if done % MANIFEST_SAVE_EVERY == 0:
                save_cleaning_manifest(output_dir, manifest)
                print(f"  Processed {skipped + done}/{total} images...")

    # Final save of the manifest
    save_cleaning_manifest(output_dir, manifest)

    print(f"\n📊 Cleaning result: {saved} bees accepted, {discarded} rejected, {skipped} already processed, out of {total} in total.")
    reason_counts = Counter(manifest["discard_reasons"].values())
    if reason_counts:
        print("Rejection reasons (all runs with these parameters):")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<12} {count}")
    if removed:
        print(f"Removed from '{output_dir}' {removed} bees previously accepted and now rejected.")

    if ORIENT_HEAD_UP:
        orient_saved_bees(output_dir, manifest, device=device, workers=workers)
    print(f"Cleaned and normalised bees saved in '{output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleans and normalises the bees extracted by extract_anomaly_dataset.py.")
    parser.add_argument("--raw", default=DATA_ROOT + "/processed/test_single_bee_batch3+4_novarroa/raw_bees")
    parser.add_argument("--clean", default=DATA_ROOT + "/processed/test_single_bee_batch3+4_novarroa/cleaned_bees")
    parser.add_argument("--device", default="auto", help="Device for the orientation (e.g. cuda:1); auto = least busy GPU")
    args = parser.parse_args()
    run_cleaning(args.raw, args.clean, device=args.device)
