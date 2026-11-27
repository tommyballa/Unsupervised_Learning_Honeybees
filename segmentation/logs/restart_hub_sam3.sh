#!/bin/bash
# After the cleaning of single_bee3+4+5+6_SAM3: embeddings, archive of the previous run, hub in phase 1 from scratch.
set -u
REPO=/home/tommaso_ballarin/bee_segmentation/bee_segmentation
AL=/mnt/disk1/tommaso_ballarin/bee_al
CLEAN=/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3/cleaned_bees
NEW_CFG=$REPO/logs/al_config_sam3_b3456.yaml
cd $REPO
echo "[$(date)] waiting for the cleaning to finish (PID 2004521)"
while kill -0 2004521 2>/dev/null; do sleep 60; done
grep -q "Pulizia terminata con codice 0" logs/cleaning_single_bee3+4+5+6_SAM3.log || { echo "Cleaning did not finish correctly: stopping"; exit 1; }
venv/bin/python - <<PY || { echo "Orientation incomplete: stopping"; exit 1; }
import json, sys
m = json.load(open("$CLEAN/_cleaning_manifest.json"))
acc = sum(s == "accepted" for s in m["processed_files"].values())
ori = len(m.get("orientation", {}).get("scores", {}))
print(f"accepted {acc}, oriented {ori}")
sys.exit(0 if acc == ori else 1)
PY

echo "[$(date)] 1/3 embedding extraction (the hub stays up on phase 2)"
venv/bin/python scripts/extract_embeddings.py --config $NEW_CFG || { echo "Embedding extraction failed: hub NOT touched"; exit 1; }

echo "[$(date)] 2/3 stopping the hub and archiving the previous run"
tmux kill-session -t active_learning
for i in $(seq 30); do pgrep -u $USER -f "python -u app.py" >/dev/null || break; sleep 1; done
pgrep -u $USER -f "python -u app.py" >/dev/null && { echo "The hub did not stop: stopping before moving the state"; exit 1; }
ARCH=$AL/archivio_$(date +%Y%m%d_%H%M%S)
mkdir -p $ARCH/config
for d in results_batch3+4 models_batch3+4 results_fase2 models_fase2; do
  [ -e $AL/$d ] && mv $AL/$d $ARCH/ && echo "  moved $AL/$d"
done
mv $REPO/data/embeddings_batch3+4 $ARCH/embeddings_batch3+4 && echo "  moved data/embeddings_batch3+4"
cp $REPO/config/al_config.yaml $REPO/config/al_fase2.yaml $ARCH/config/
cat > $ARCH/LEGGIMI.txt <<TXT
Active learning run on batch3+4 (phase 1 segmentation quality + phase 2 profile/anomalies),
archived on $(date) before restarting from scratch on single_bee3+4+5+6_SAM3.
Original paths:
  results_batch3+4, models_batch3+4, results_fase2, models_fase2 -> $AL/
  embeddings_batch3+4 -> $REPO/data/
  config/al_config.yaml (phase 1), config/al_fase2.yaml (phase 2) -> $REPO/config/
To reuse it: put the folders back in their original paths and the configs in config/.
TXT
cp $NEW_CFG $REPO/config/al_config.yaml

echo "[$(date)] 3/3 starting the hub in phase 1 ($ARCH)"
tmux new-session -d -s active_learning -c $REPO/active_learning_tool \
  "env -u BEE_AL_CONFIG ../venv/bin/python -u app.py 2>&1 | tee -a ../logs/hub_app.log"
for i in $(seq 120); do curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/ | grep -qE "200|302" && break; sleep 5; done
echo "[$(date)] hub responds: HTTP $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/)"
