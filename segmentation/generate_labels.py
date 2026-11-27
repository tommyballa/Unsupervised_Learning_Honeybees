from bee_paths import DATA_ROOT
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision
from lang_sam import LangSAM
import gc

# init model
model = LangSAM()

datasets = [
    (DATA_ROOT + "/processed/train_batch4_labelled/images", DATA_ROOT + "/processed/train_batch4_labelled/labels"),
    (DATA_ROOT + "/processed/test_batch4_labelled/images", DATA_ROOT + "/processed/test_batch4_labelled/labels")
]

text_prompt = "bee."
TARGET_BEE_SIZE = 120.0

for image_dir, label_dir in datasets:
    print(f"Processing dataset: {image_dir}")
    os.makedirs(label_dir, exist_ok=True)
    
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        if os.path.exists(label_path):
            print(f"Skipping {img_name} (already processed)")
            continue
            
        img_path = os.path.join(image_dir, img_name)
        original_pil = Image.open(img_path).convert("RGB")
        
        # pass 1: estimate bee size
        scan_pil = original_pil.copy()
        scan_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        ratio_w = original_pil.width / scan_pil.width
        ratio_h = original_pil.height / scan_pil.height
        scale_factor = 1.0
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                try:
                    quick_results = model.predict(
                        [scan_pil], 
                        [text_prompt], 
                        box_threshold=0.25, 
                        text_threshold=0.25
                    )
                except Exception as e:
                    print(f"Error during quick scan: {e}")
                    quick_results = []
                
        if len(quick_results) > 0 and len(quick_results[0].get("boxes", [])) > 0:
            boxes = quick_results[0].get("boxes", [])
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
                
            sizes = []
            for box in boxes:
                w = (box[2] - box[0]) * ratio_w
                h = (box[3] - box[1]) * ratio_h
                sizes.append(max(w, h)) 
                
            median_bee_size = float(np.median(sizes))
            
            if median_bee_size > 0:
                scale_factor = TARGET_BEE_SIZE / median_bee_size
                scale_factor = max(0.3, min(scale_factor, 5.0)) 
                
        try: del quick_results, boxes
        except NameError: pass
        gc.collect()
        torch.cuda.empty_cache()

        # pass 2: apply zoom and tile overlapping
        new_w = int(original_pil.width * scale_factor)
        new_h = int(original_pil.height * scale_factor)
        image_pil = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_w, img_h = image_pil.size
        
        TILE_SIZE = 500
        # 40% overlap ensures bees discarded on borders are fully visible in the adjacent tile
        OVERLAP = int(TILE_SIZE * 0.40) 
        STRIDE = TILE_SIZE - OVERLAP
        
        crops = []
        for y in range(0, img_h, STRIDE):
            for x in range(0, img_w, STRIDE):
                x1 = x
                y1 = y
                x2 = min(img_w, x + TILE_SIZE)
                y2 = min(img_h, y + TILE_SIZE)
                
                if x2 - x1 < 50 or y2 - y1 < 50:
                    continue
                    
                crops.append((x1, y1, x2, y2))
                
        global_contours = []
        global_boxes = []
        global_scores = []
        
        for crop_box in crops:
            x1, y1, x2, y2 = crop_box
            tile = image_pil.crop(crop_box)
            
            torch.cuda.empty_cache()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    try:
                        result = model.predict(
                            [tile], 
                            [text_prompt], 
                            box_threshold=0.25, 
                            text_threshold=0.25
                        )
                    except Exception:
                        continue
            
            if not result or len(result) == 0:
                continue
                
            masks = result[0].get("masks", [])
            scores = result[0].get("scores", [])
            
            for m_idx, mask_tensor in enumerate(masks):
                score = float(scores[m_idx]) if len(scores) > m_idx else 1.0
                
                if hasattr(mask_tensor, "cpu"):
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = (mask_tensor * 255).astype(np.uint8)
                    
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 600: 
                        continue
                    
                    tile_area = (x2 - x1) * (y2 - y1)
                    if area > 0.25 * tile_area:
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
                        
                    # shift to global coordinates
                    shifted_contour = contour + np.array([x1, y1])
                    
                    bx, by, bw, bh = cv2.boundingRect(shifted_contour)
                    box = [bx, by, bx + bw, by + bh]
                    
                    global_contours.append(shifted_contour)
                    global_boxes.append(box)
                    global_scores.append(score)
            
            try: del result, masks, scores, tile
            except NameError: pass
            gc.collect()

        # pass 3: NMS and save
        if len(global_boxes) > 0:
            boxes_tensor = torch.tensor(global_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(global_scores, dtype=torch.float32)
            
            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            
            with open(label_path, 'w') as f:
                for idx in keep_indices:
                    idx = int(idx.item())
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
                    
        gc.collect()
        torch.cuda.empty_cache()