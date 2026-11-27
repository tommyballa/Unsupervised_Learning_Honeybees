#!/bin/bash
# Foto del telefono (batch_6/favi_phone_camera, 22 foto mai viste dal YOLO): estrazione con lo stesso YOLO img1376 a 2048 px di batch 3-6, poi pulizia v4 + testa in alto.
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
# aspetta che finisca la valutazione sulla stessa GPU
while tmux has-session -t phone_eval 2>/dev/null; do sleep 30; done
D=/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee_phone_SAM3
source venv/bin/activate
echo "Inizio estrazione ($(date))"
CUDA_VISIBLE_DEVICES=0 python -u extract_anomaly_dataset.py \
  --model runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt \
  --source /home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/batch_6/favi_phone_camera \
  --out "$D/raw_bees" --imgsz 2048 --device 0 || { echo "Estrazione fallita"; exit 1; }
n=$(python3 -c "import json;print(len(json.load(open('$D/raw_bees/_extraction_manifest.json'))['processed_sources']))")
echo "Estrazione terminata: $n/22 foto nel manifest ($(date))"
if [ "$n" -ne 22 ]; then echo "Estrazione incompleta, pulizia NON lanciata"; exit 1; fi
CUDA_VISIBLE_DEVICES=0 python -u dataset_cleaning.py --raw "$D/raw_bees" --clean "$D/cleaned_bees"
echo "Pulizia terminata con codice $? ($(date))"
