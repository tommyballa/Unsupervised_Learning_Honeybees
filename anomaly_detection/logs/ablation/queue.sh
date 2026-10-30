#!/bin/bash
# Ablation single-bee (numero di api di train, backbone): vedi experiments/single_bee/ablation.py.
# Uscite in checkpoints|heatmaps/ablation/<modello>/<tag>, log in logs/ablation/.
# Più corsie (anche su GPU diverse) pescano dalla stessa lista di job: ogni job
# viene preso da una sola corsia (lock in logs/ablation/locks/).
# Uso: GPU=<idx> bash logs/ablation/queue.sh logs/ablation/jobs.txt
#   una riga per job: <modello>:n<N> (ABL_TRAIN_N=N) | <modello>:<backbone> (ABL_BACKBONE)
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
PY=.venv/bin/python
mkdir -p logs/ablation/locks
for job in $(grep -v '^#' "$1"); do
  IFS=: read -r model spec <<< "$job"
  if [[ $spec =~ ^n[0-9]+$ ]]; then N=${spec#n}; BB=; else N=; BB=$spec; fi
  tag=${model}_$spec
  [ -e logs/ablation/$tag.done ] && continue
  mkdir logs/ablation/locks/$tag 2>/dev/null || continue
  echo "$(date '+%F %T') GPU$GPU start $tag" >> logs/ablation/progress.txt
  export ABL_TRAIN_N=$N ABL_BACKBONE=$BB CUDA_VISIBLE_DEVICES=$GPU
  rm -f logs/ablation/$tag.failed
  $PY -u experiments/single_bee/main_$model.py > logs/ablation/${tag}_main.log 2>&1 \
    && $PY -u experiments/plot_heatmaps.py single_bee $model > logs/ablation/${tag}_plot.log 2>&1 \
    && touch logs/ablation/$tag.done || touch logs/ablation/$tag.failed
  echo "$(date '+%F %T') GPU$GPU end $tag ($(ls logs/ablation/$tag.done logs/ablation/$tag.failed 2>/dev/null | xargs -n1 basename))" >> logs/ablation/progress.txt
done
