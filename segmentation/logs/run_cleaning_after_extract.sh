#!/bin/bash
# Waits for the extraction (PID 4135013) to finish, then runs the cleaning of the single_bee3+4+5+6_SAM3 dataset
cd "$(dirname "$0")/.."   # segmentation/
D=$BEES_DATA_ROOT/processed/single_bee3+4+5+6_SAM3
while kill -0 4135013 2>/dev/null; do sleep 60; done
n=$(python3 -c "import json;print(len(json.load(open('$D/raw_bees/_extraction_manifest.json'))['processed_sources']))")
echo "Extraction finished: $n/791 photos in the manifest ($(date))"
if [ "$n" -ne 791 ]; then echo "Extraction incomplete, cleaning NOT started"; tail -20 logs/extract_single_bee3+4+5+6_SAM3.log; exit 1; fi
source venv/bin/activate
python dataset_cleaning.py --raw "$D/raw_bees" --clean "$D/cleaned_bees"
echo "Cleaning finished with code $? ($(date))"
