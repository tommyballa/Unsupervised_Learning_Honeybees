#!/bin/bash
# Dopo la pulizia di single_bee3+4+5+6_SAM3: embedding, archivio della run precedente, hub in fase 1 da zero.
set -u
REPO=/home/tommaso_ballarin/bee_segmentation/bee_segmentation
AL=/mnt/disk1/tommaso_ballarin/bee_al
CLEAN=/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/single_bee3+4+5+6_SAM3/cleaned_bees
NEW_CFG=$REPO/logs/al_config_sam3_b3456.yaml
cd $REPO
echo "[$(date)] attendo la fine della pulizia (PID 2004521)"
while kill -0 2004521 2>/dev/null; do sleep 60; done
grep -q "Pulizia terminata con codice 0" logs/cleaning_single_bee3+4+5+6_SAM3.log || { echo "Pulizia non terminata correttamente: mi fermo"; exit 1; }
venv/bin/python - <<PY || { echo "Orientamento incompleto: mi fermo"; exit 1; }
import json, sys
m = json.load(open("$CLEAN/_cleaning_manifest.json"))
acc = sum(s == "accepted" for s in m["processed_files"].values())
ori = len(m.get("orientation", {}).get("scores", {}))
print(f"accettate {acc}, orientate {ori}")
sys.exit(0 if acc == ori else 1)
PY

echo "[$(date)] 1/3 estrazione embedding (l'hub resta acceso sulla fase 2)"
venv/bin/python scripts/extract_embeddings.py --config $NEW_CFG || { echo "Estrazione embedding fallita: hub NON toccato"; exit 1; }

echo "[$(date)] 2/3 stop hub e archivio della run precedente"
tmux kill-session -t active_learning
for i in $(seq 30); do pgrep -u $USER -f "python -u app.py" >/dev/null || break; sleep 1; done
pgrep -u $USER -f "python -u app.py" >/dev/null && { echo "L'hub non si è fermato: mi fermo prima di spostare lo stato"; exit 1; }
ARCH=$AL/archivio_$(date +%Y%m%d_%H%M%S)
mkdir -p $ARCH/config
for d in results_batch3+4 models_batch3+4 results_fase2 models_fase2; do
  [ -e $AL/$d ] && mv $AL/$d $ARCH/ && echo "  spostato $AL/$d"
done
mv $REPO/data/embeddings_batch3+4 $ARCH/embeddings_batch3+4 && echo "  spostato data/embeddings_batch3+4"
cp $REPO/config/al_config.yaml $REPO/config/al_fase2.yaml $ARCH/config/
cat > $ARCH/LEGGIMI.txt <<TXT
Run di active learning su batch3+4 (fase 1 qualità segmentazione + fase 2 profilo/anomalie),
archiviata il $(date) prima di ripartire da zero su single_bee3+4+5+6_SAM3.
Percorsi originali:
  results_batch3+4, models_batch3+4, results_fase2, models_fase2 -> $AL/
  embeddings_batch3+4 -> $REPO/data/
  config/al_config.yaml (fase 1), config/al_fase2.yaml (fase 2) -> $REPO/config/
Per riusarla: rimettere le cartelle nei percorsi originali e le config in config/.
TXT
cp $NEW_CFG $REPO/config/al_config.yaml

echo "[$(date)] 3/3 avvio hub in fase 1 ($ARCH)"
tmux new-session -d -s active_learning -c $REPO/active_learning_tool \
  "env -u BEE_AL_CONFIG ../venv/bin/python -u app.py 2>&1 | tee -a ../logs/hub_app.log"
for i in $(seq 120); do curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/ | grep -qE "200|302" && break; sleep 5; done
echo "[$(date)] hub risponde: HTTP $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/)"
