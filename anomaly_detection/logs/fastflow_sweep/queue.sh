#!/bin/bash
# uso: queue.sh <file con una run per riga: tag args...>
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
while read -r tag args; do
  [ -z "$tag" ] && continue
  [ -f logs/fastflow_sweep/$tag.done ] && continue
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/single_bee/sweep_fastflow.py --tag $tag $args > logs/fastflow_sweep/$tag.log 2>&1 && touch logs/fastflow_sweep/$tag.done
done < "$1"
