#!/bin/bash
# Re-run single-bee con validation set (best e soglie su val/, metriche su test_heldout/).
# Uscite in checkpoints/single_bee_val, heatmaps/single_bee_val, logs/val.
# Uso: GPU=<idx> bash logs/val/queue.sh job1 job2 ...
#   job = <modello> (training + plot) | eval:<modello>[:<backbone patchcore>] (solo plot)
cd "$(dirname "$0")/../.."   # anomaly_detection/
PY=.venv/bin/python
for job in "$@"; do
  IFS=: read -r kind model backbone <<< "$job"
  if [ "$kind" = eval ]; then
    tag=$model${backbone:+_$backbone}
    echo "$(date '+%F %T') GPU$GPU start eval $tag" >> logs/val/progress.txt
    PATCHCORE_BACKBONE=$backbone CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/plot_heatmaps.py single_bee $model \
      > logs/val/${tag}_plot.log 2>&1 && touch logs/val/$tag.done || touch logs/val/$tag.failed
  else
    tag=$kind
    echo "$(date '+%F %T') GPU$GPU start train $tag" >> logs/val/progress.txt
    CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/single_bee/main_$tag.py > logs/val/${tag}_main.log 2>&1 \
      && CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/plot_heatmaps.py single_bee $tag > logs/val/${tag}_plot.log 2>&1 \
      && touch logs/val/$tag.done || touch logs/val/$tag.failed
  fi
  echo "$(date '+%F %T') GPU$GPU end $tag ($(ls logs/val/$tag.done logs/val/$tag.failed 2>/dev/null | xargs -n1 basename))" >> logs/val/progress.txt
done
