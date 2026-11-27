#!/bin/bash
# Phone photos (batch_6/favi_phone_camera, 22 photos never seen by YOLO): extraction with the same YOLO img1376 at 2048 px as batches 3-6, then cleaning v4 + head at the top.
cd "$(dirname "$0")/../.."   # segmentation/
# wait for the evaluation on the same GPU to finish
while tmux has-session -t phone_eval 2>/dev/null; do sleep 30; done
D=$BEES_DATA_ROOT/processed/single_bee_phone_SAM3
source venv/bin/activate
echo "Starting extraction ($(date))"
CUDA_VISIBLE_DEVICES=0 python -u extract_anomaly_dataset.py \
  --model runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt \
  --source $BEES_DATA_ROOT/batch_6/favi_phone_camera \
  --out "$D/raw_bees" --imgsz 2048 --device 0 || { echo "Extraction failed"; exit 1; }
n=$(python3 -c "import json;print(len(json.load(open('$D/raw_bees/_extraction_manifest.json'))['processed_sources']))")
echo "Extraction finished: $n/22 photos in the manifest ($(date))"
if [ "$n" -ne 22 ]; then echo "Extraction incomplete, cleaning NOT launched"; exit 1; fi
CUDA_VISIBLE_DEVICES=0 python -u dataset_cleaning.py --raw "$D/raw_bees" --clean "$D/cleaned_bees"
echo "Cleaning finished with code $? ($(date))"
