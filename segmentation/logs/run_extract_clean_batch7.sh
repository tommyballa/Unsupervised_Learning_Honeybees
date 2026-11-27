#!/bin/bash
# Batch 7 (validation): extraction with the same YOLO img1376 at 2048 px as batches 3-6, then v4 cleaning + head-up orientation.
cd "$(dirname "$0")/.."   # segmentation/
D=$BEES_DATA_ROOT/processed/single_bee7_SAM3
source venv/bin/activate
echo "Extraction started ($(date))"
CUDA_VISIBLE_DEVICES=1 python -u extract_anomaly_dataset.py \
  --model runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt \
  --source $BEES_DATA_ROOT/batch_7/favi \
  --out "$D/raw_bees" --imgsz 2048 --device 0 || { echo "Extraction failed"; exit 1; }
n=$(python3 -c "import json;print(len(json.load(open('$D/raw_bees/_extraction_manifest.json'))['processed_sources']))")
echo "Extraction finished: $n/263 photos in the manifest ($(date))"
if [ "$n" -ne 263 ]; then echo "Extraction incomplete, cleaning NOT started"; exit 1; fi
CUDA_VISIBLE_DEVICES=1 python -u dataset_cleaning.py --raw "$D/raw_bees" --clean "$D/cleaned_bees"
echo "Cleaning finished with code $? ($(date))"
