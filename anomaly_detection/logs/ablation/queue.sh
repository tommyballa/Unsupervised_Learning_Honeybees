#!/bin/bash
# Single-bee ablation (number of training bees, backbone): see experiments/single_bee/ablation.py.
# Outputs in checkpoints|heatmaps/ablation/<model>/<tag>, logs in logs/ablation/.
# Several lanes (even on different GPUs) draw from the same job list: each job
# is taken by a single lane (locks in logs/ablation/locks/).
# Usage: GPU=<idx> bash logs/ablation/queue.sh logs/ablation/jobs.txt
#   one line per job: <model>:n<N> (ABL_TRAIN_N=N) | <model>:<backbone> (ABL_BACKBONE)
cd "$(dirname "$0")/../.."   # anomaly_detection/
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
