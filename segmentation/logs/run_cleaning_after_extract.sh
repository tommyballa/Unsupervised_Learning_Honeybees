#!/bin/bash
# Aspetta la fine dell'estrazione (PID 4135013) e poi lancia la pulizia del dataset single_bee3+4+5+6_SAM3
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
D=/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3
while kill -0 4135013 2>/dev/null; do sleep 60; done
n=$(python3 -c "import json;print(len(json.load(open('$D/raw_bees/_extraction_manifest.json'))['processed_sources']))")
echo "Estrazione terminata: $n/791 foto nel manifest ($(date))"
if [ "$n" -ne 791 ]; then echo "Estrazione incompleta, pulizia NON lanciata"; tail -20 logs/extract_single_bee3+4+5+6_SAM3.log; exit 1; fi
source venv/bin/activate
python dataset_cleaning.py --raw "$D/raw_bees" --clean "$D/cleaned_bees"
echo "Pulizia terminata con codice $? ($(date))"
