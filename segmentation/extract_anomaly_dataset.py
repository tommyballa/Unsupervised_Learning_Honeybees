from bee_paths import DATA_ROOT
import os
import argparse
import cv2
import glob
import json
import numpy as np
from ultralytics import YOLO

# Name of the manifest file that tracks the source images already processed
MANIFEST_FILENAME = "_extraction_manifest.json"

# ============================================================
# PER-PHOTO NORMALISATION (brightness and contrast)
# ============================================================
# Before cropping, the photo is brought to a common brightness and contrast with a
# linear transformation of the L channel (LAB): a single gain and a single offset for
# the whole photo, so the bees of the same photo remain comparable with each other.
# The statistics are measured on the bee pixels only, because the background (comb,
# wood, frame) changes from photo to photo. The targets are the medians measured on 90
# train/test photos of batch3+4 (mean L 72-139, std 33-51 before the correction).
NORMALIZE_PHOTO = True
TARGET_L_MEAN = 110.0       # Mean brightness of the bees after normalisation (L in 0-255)
TARGET_L_STD = 42.0         # Contrast (standard deviation of L) of the bees after normalisation
GAIN_RANGE = (0.6, 1.8)     # Gain limits, so as not to amplify the noise in flat photos
MIN_BEE_PIXELS = 5000       # Below this threshold the statistics are measured on the whole photo


def load_manifest(out_dir):
    """Loads the manifest of the source images already processed."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {"processed_sources": {}, "next_bee_id": 0}


def save_manifest(out_dir, manifest):
    """Saves the updated manifest to disk."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def photo_normalization_params():
    """Parameters that determine the saved pixels: bees extracted with different parameters must not be mixed."""
    if not NORMALIZE_PHOTO:
        return {"enabled": False}
    return {"enabled": True, "target_l_mean": TARGET_L_MEAN, "target_l_std": TARGET_L_STD,
            "gain_range": list(GAIN_RANGE), "min_bee_pixels": MIN_BEE_PIXELS}


def normalize_photo(img_bgr, bee_mask):
    """
    Brings the brightness and contrast of the photo to the target values, measuring them on the bee pixels.

    Returns:
        (normalised image, gain, offset) with L' = L * gain + offset
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
    parser = argparse.ArgumentParser(description="Extracts the single bees segmented by YOLO as RGBA PNGs.")
    parser.add_argument("--model", default="runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4_img1440/weights/best.pt")
    # Folder to extract the bees from
    parser.add_argument("--source", default=DATA_ROOT + "/processed/test_batch3+4/images")
    # Output: folder with ALL the raw extracted bees (no filters)
    parser.add_argument("--out", default=DATA_ROOT + "/processed/test_single_bee_batch3+4_novarroa/raw_bees")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default=None, help="E.g. 0 or cpu; default: automatic choice by ultralytics")
    return parser.parse_args()


def main():
    args = parse_args()
    print(" Loading the YOLO model...")
    model = YOLO(args.model)
    source_dir = args.source
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the manifest to know which source images have already been processed
    manifest = load_manifest(out_dir)

    # Bees already extracted must have the same normalisation as the new ones
    norm_params = photo_normalization_params()
    previous_params = manifest.get("photo_normalization", {"enabled": False})
    if manifest["processed_sources"] and previous_params != norm_params:
        raise SystemExit(
            f"'{out_dir}' contains bees extracted with a different normalisation ({previous_params}).\n"
            f"Use a new output folder or empty the existing one so they are not mixed.")
    manifest["photo_normalization"] = norm_params
    
    print(f"Extracting bees from: {source_dir}")
    
    image_paths = (
        glob.glob(os.path.join(source_dir, "*.[jJ][pP][gG]")) +
        glob.glob(os.path.join(source_dir, "*.[pP][nN][gG]"))
    )
    
    bee_counter = manifest["next_bee_id"]
    skipped = 0
    extracted = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Skip if this source image was already processed in a previous run
        if img_name in manifest["processed_sources"]:
            skipped += 1
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Run inference at maximum resolution. retina_masks=True greatly improves the edges.
        results = model.predict(source=img, save=False, show=False, conf=args.conf, imgsz=args.imgsz,
                                max_det=2000, device=args.device, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            manifest["processed_sources"][img_name] = {"bees": 0}
            save_manifest(out_dir, manifest)
            continue
            
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        # Per-photo normalisation: inference has already been run on the original photo,
        # from here on the crops are taken from the normalised photo.
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
            
            # Create a black mask the size of the original image
            blank_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Take the polygon coordinates of this specific bee
            polygon = res.masks.xy[i]
            if len(polygon) == 0:
                continue
            
            # Draw the polygon filled in white on the black mask
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(blank_mask, [pts], 255)
            
            # ---------------------------------------------------------------
            # NOTE: the mask refinement operations (morphological
            # closing, dilation, GaussianBlur+threshold) are NO longer
            # performed here. They are centralised in dataset_cleaning.py
            # to avoid duplication and preserve the original quality.
            # ---------------------------------------------------------------
            
            # Crop with generous padding around the bounding box
            padding = 20
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(w, x2 + padding)
            y2_p = min(h, y2 + padding)
            
            # Crop both the photo and the mask using the enlarged coordinates
            img_crop = img[y1_p:y2_p, x1_p:x2_p].copy()
            mask_crop = blank_mask[y1_p:y2_p, x1_p:x2_p]
            
            # ---------------------------------------------------------------
            # Save as RGBA PNG: the BGR channels contain the original,
            # unmodified image, the Alpha channel contains the raw mask of the
            # YOLO polygon. This preserves both the original pixels and the
            # mask without any loss of quality.
            #
            # We NO longer resize to 512x512 here: the single resize
            # happens in dataset_cleaning.py after all the filters, avoiding
            # the double resize that degraded the image.
            # ---------------------------------------------------------------
            b_ch, g_ch, r_ch = cv2.split(img_crop)
            rgba = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
            
            # Save the result with the confidence in the name for debugging (lossless PNG with zero compression for maximum fidelity)
            bee_filename = f"bee_{bee_counter:05d}_conf{confs[i]:.2f}.png"
            cv2.imwrite(os.path.join(out_dir, bee_filename), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            bee_counter += 1
            bees_from_this_image += 1
            extracted += 1
        
        # Update the manifest after each source image, to be robust to interruptions
        manifest["processed_sources"][img_name] = {"bees": bees_from_this_image, **photo_stats}
        manifest["next_bee_id"] = bee_counter
        save_manifest(out_dir, manifest)
            
    print(f"\n✅ Extraction complete! Extracted {extracted} new bees ({skipped} source images already processed, skipped).")
    print(f"Total bees in the folder: {bee_counter}")
    print(f"Raw bees saved in '{out_dir}'.")
    print("Now run dataset_cleaning.py to clean and normalise the dataset.")

if __name__ == "__main__":
    main()
