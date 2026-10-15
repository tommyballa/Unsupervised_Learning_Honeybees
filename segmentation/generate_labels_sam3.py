import os
import shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
import torchvision
from transformers import Sam3Model, Sam3Processor
import gc

# init model (gated repo: requires `hf auth login` with access to facebook/sam3)
MODEL_ID = "facebook/sam3"
processor = Sam3Processor.from_pretrained(MODEL_ID)
model = Sam3Model.from_pretrained(MODEL_ID).to("cuda").eval()

# source batches, searched recursively (batch_6 has both favi/ and sick/)
BASE_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi"
source_dirs = [
    os.path.join(BASE_DIR, "batch_3"),
    os.path.join(BASE_DIR, "batch_4"),
    os.path.join(BASE_DIR, "batch_5"),
    os.path.join(BASE_DIR, "batch_6"),
]

# single output dataset: every labelled image is copied next to its labels
OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "bee_batch3+4+5+6_SAM3")
image_out_dir = os.path.join(OUTPUT_DIR, "images")
label_dir = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

text_prompt = "bee"
MASK_THRESHOLD = 0.5
BATCH_SIZE = 8                  # tiles sent to the model together

# --- parameters tuned for recall on 18 photos spanning zoom (bees 150-680 px) and focus ---

# scale estimate: the photo is split in a SCAN_GRID x SCAN_GRID grid of crops. On the whole-photo
# thumbnail SAM 3 sees no bees at all in the wide, crowded shots (59 of 791 photos had 0-5 boxes).
SCAN_GRID = 3
SCAN_SCORE_THRESHOLD = 0.5

# the photo is tiled once per target size and the results are merged: a single pass leaves about
# 1% of the bees without a label, two passes at different zoom about 0.25%
TARGET_BEE_SIZES = [120.0, 150.0]   # bee length in px inside the 500 px tile
SCALE_RANGE = (0.05, 5.0)           # the old 0.3 lower bound left close-up bees larger than the tile overlap
TILE_SIZE = 500
OVERLAP_RATIO = 0.40

# scores: below 0.5 SAM 3 returns almost only heads of buried bees, pollen cells and gaps, so a
# low-score mask is kept only when it is about as large as a whole bee. Above 0.5 everything is
# kept: raising the threshold to 0.6 loses about 8% of the bees.
SCORE_THRESHOLD = 0.5
LOW_SCORE_THRESHOLD = 0.3
LOW_SCORE_MIN_REL_AREA = 0.6        # area relative to the median bee of the photo

MIN_AREA = 600                      # px^2 inside the tile
MAX_AREA_TILE_RATIO = 0.25

# NMS on the masks instead of the boxes: a mask is dropped when more than this fraction of the
# smaller mask lies inside a higher-scored one. Box NMS left about 5% duplicates and fragments.
NMS_MASK_OVERLAP = 0.6


def predict(images, score_threshold):
    """Run SAM 3 on a list of images, returns one (masks, scores, boxes) tuple of numpy arrays per image."""
    results = []
    for start in range(0, len(images), BATCH_SIZE):
        chunk = images[start:start + BATCH_SIZE]
        inputs = processor(images=chunk, text=[text_prompt] * len(chunk), return_tensors="pt").to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                outputs = model(**inputs)
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=inputs.get("original_sizes").tolist()
        )
        for result in processed:
            results.append((
                result["masks"].cpu().numpy() > 0,
                result["scores"].float().cpu().numpy(),
                result["boxes"].float().cpu().numpy(),
            ))
        del inputs, outputs, processed
    return results


def estimate_bee_size(original_pil):
    """Median bee length in original pixels, measured on a grid of crops. None if no bee is found."""
    img_w, img_h = original_pil.size
    crops = []
    for gy in range(SCAN_GRID):
        for gx in range(SCAN_GRID):
            crops.append(original_pil.crop((gx * img_w // SCAN_GRID, gy * img_h // SCAN_GRID,
                                            (gx + 1) * img_w // SCAN_GRID, (gy + 1) * img_h // SCAN_GRID)))
    ratios, scans = [], []
    for crop in crops:
        scan = crop.copy()
        scan.thumbnail((1008, 1008), Image.Resampling.LANCZOS)
        ratios.append(crop.width / scan.width)
        scans.append(scan)

    all_sizes, inner_sizes = [], []
    for crop, ratio, (_, _, boxes) in zip(crops, ratios, predict(scans, SCAN_SCORE_THRESHOLD)):
        for box in boxes * ratio:
            size = float(max(box[2] - box[0], box[3] - box[1]))
            all_sizes.append(size)
            # bees chopped by the crop edge look shorter than they are
            if box[0] > 4 and box[1] > 4 and box[2] < crop.width - 4 and box[3] < crop.height - 4:
                inner_sizes.append(size)

    sizes = inner_sizes if len(inner_sizes) >= 5 else all_sizes
    return float(np.median(sizes)) if sizes else None


def tile_candidates(original_pil, scale_factor):
    """Tile the rescaled photo and return (contour in original pixels, score, area in original px^2)."""
    new_w = int(original_pil.width * scale_factor)
    new_h = int(original_pil.height * scale_factor)
    image_pil = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_w, img_h = image_pil.size
    to_original = np.array([original_pil.width / img_w, original_pil.height / img_h])

    # the overlap ensures bees discarded on borders are fully visible in the adjacent tile
    stride = TILE_SIZE - int(TILE_SIZE * OVERLAP_RATIO)

    crops = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x1 = x
            y1 = y
            x2 = min(img_w, x + TILE_SIZE)
            y2 = min(img_h, y + TILE_SIZE)

            if x2 - x1 < 50 or y2 - y1 < 50:
                continue

            crops.append((x1, y1, x2, y2))

    candidates = []
    # one batch at a time, so only BATCH_SIZE tiles of masks are in memory together
    for start in range(0, len(crops), BATCH_SIZE):
        chunk = crops[start:start + BATCH_SIZE]
        try:
            predictions = predict([image_pil.crop(crop_box) for crop_box in chunk], LOW_SCORE_THRESHOLD)
        except Exception as e:
            print(f"Error on a tile batch: {e}")
            continue

        for (x1, y1, x2, y2), (masks, scores, _) in zip(chunk, predictions):
            for m_idx, mask in enumerate(masks):
                score = float(scores[m_idx]) if len(scores) > m_idx else 1.0

                mask_np = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < MIN_AREA:
                        continue

                    tile_area = (x2 - x1) * (y2 - y1)
                    if area > MAX_AREA_TILE_RATIO * tile_area:
                        continue

                    # border check: discard instances chopped by the tile edge
                    flat_contour = contour.flatten()
                    xs = flat_contour[0::2]
                    ys = flat_contour[1::2]

                    touches_border = False
                    if x1 > 0 and np.min(xs) <= 2: touches_border = True
                    if x2 < img_w and np.max(xs) >= (x2 - x1) - 3: touches_border = True
                    if y1 > 0 and np.min(ys) <= 2: touches_border = True
                    if y2 < img_h and np.max(ys) >= (y2 - y1) - 3: touches_border = True

                    if touches_border:
                        continue

                    # shift to global coordinates, then back to the original resolution so that
                    # the passes at different zoom share the same frame
                    shifted_contour = (contour + np.array([x1, y1])) * to_original
                    candidates.append((np.round(shifted_contour).astype(np.int32), score,
                                       area * to_original[0] * to_original[1]))

        del predictions
        torch.cuda.empty_cache()

    return candidates


def mask_nms(contours, scores, raster_scale):
    """Greedy NMS on the masks: returns the indices kept, best score first.

    A candidate is dropped when more than NMS_MASK_OVERLAP of the smaller of the two masks is
    shared with an already kept one. This removes the same bee seen in several tiles and the
    fragments (a wing, a head) that lie inside a bee already labelled.
    """
    boxes, masks, areas = [], [], []
    for contour in contours:
        pts = np.round(contour.reshape(-1, 2) * raster_scale).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(pts)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts - np.array([bx, by])], 1)
        boxes.append([bx, by, bx + bw, by + bh])
        masks.append(mask.astype(bool))
        areas.append(int(mask.sum()))

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    boxes_overlap = torchvision.ops.box_iou(boxes_tensor, boxes_tensor).numpy() > 0
    order = np.argsort(-np.asarray(scores), kind="stable")

    kept = np.zeros(len(contours), dtype=bool)
    keep_indices = []
    for i in order:
        suppressed = False
        for j in np.nonzero(boxes_overlap[i] & kept)[0]:
            ix1, iy1 = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
            ix2, iy2 = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            part_i = masks[i][iy1 - boxes[i][1]:iy2 - boxes[i][1], ix1 - boxes[i][0]:ix2 - boxes[i][0]]
            part_j = masks[j][iy1 - boxes[j][1]:iy2 - boxes[j][1], ix1 - boxes[j][0]:ix2 - boxes[j][0]]
            intersection = int(np.logical_and(part_i, part_j).sum())
            if intersection > NMS_MASK_OVERLAP * max(1, min(areas[i], areas[j])):
                suppressed = True
                break
        if not suppressed:
            kept[i] = True
            keep_indices.append(int(i))
    return keep_indices


for source_dir in source_dirs:
    print(f"Processing batch: {source_dir}")

    image_paths = []
    for root, _, files in os.walk(source_dir):
        for name in sorted(files):
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, name))

    for img_path in sorted(image_paths):
        img_name = os.path.basename(img_path)

        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        if os.path.exists(label_path):
            print(f"Skipping {img_name} (already processed)")
            continue

        # apply the EXIF orientation like cv2 (and so YOLO) does: some phone photos are stored
        # upside down, and their labels would be rotated 180° with respect to the training image
        original_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

        # pass 1: estimate bee size
        try:
            median_bee_size = estimate_bee_size(original_pil)
        except Exception as e:
            print(f"Error during quick scan: {e}")
            median_bee_size = None

        gc.collect()
        torch.cuda.empty_cache()

        # pass 2: apply zoom and tile overlapping, once per target bee size
        global_contours = []
        global_scores = []
        global_areas = []

        for target_bee_size in TARGET_BEE_SIZES:
            scale_factor = 1.0
            if median_bee_size is not None and median_bee_size > 0:
                scale_factor = target_bee_size / median_bee_size
                scale_factor = max(SCALE_RANGE[0], min(scale_factor, SCALE_RANGE[1]))

            for contour, score, area in tile_candidates(original_pil, scale_factor):
                global_contours.append(contour)
                global_scores.append(score)
                global_areas.append(area)

            gc.collect()

            # without a size estimate every pass would run at the same scale
            if median_bee_size is None:
                break

        # pass 3: NMS and save
        final_indices = []
        if len(global_contours) > 0:
            # masks are rasterized with bees about 120 px long, whatever the photo resolution
            raster_scale = min(1.0, 120.0 / median_bee_size) if median_bee_size else 1.0
            keep_indices = mask_nms(global_contours, global_scores, raster_scale)

            confident_areas = [global_areas[i] for i in keep_indices if global_scores[i] >= SCORE_THRESHOLD]
            median_area = float(np.median(confident_areas)) if confident_areas else 0.0

            for idx in keep_indices:
                if global_scores[idx] >= SCORE_THRESHOLD:
                    final_indices.append(idx)
                elif median_area > 0 and global_areas[idx] >= LOW_SCORE_MIN_REL_AREA * median_area:
                    final_indices.append(idx)

        if len(final_indices) > 0:
            img_w, img_h = original_pil.size

            with open(label_path, 'w') as f:
                for idx in final_indices:
                    contour = global_contours[idx]

                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    flat_contour = approx.flatten()

                    normalized_points = []
                    for p_i in range(len(flat_contour)):
                        if p_i % 2 == 0:
                            normalized_points.append(max(0.0, min(1.0, flat_contour[p_i] / img_w)))
                        else:
                            normalized_points.append(max(0.0, min(1.0, flat_contour[p_i] / img_h)))

                    points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                    f.write(f"0 {points_str}\n")

            shutil.copy2(img_path, os.path.join(image_out_dir, img_name))
            print(f"{img_name}: {len(final_indices)} bees")
        else:
            print(f"No bees found in {img_name}, not added to the dataset")

        gc.collect()
        torch.cuda.empty_cache()
