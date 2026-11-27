#!/bin/bash
# usage: queue.sh <file with one run per line: tag args...>
cd "$(dirname "$0")/../.."   # anomaly_detection/
while read -r tag args; do
  [ -z "$tag" ] && continue
  [ -f logs/fastflow_sweep/$tag.done ] && continue
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/single_bee/sweep_fastflow.py --tag $tag $args > logs/fastflow_sweep/$tag.log 2>&1 && touch logs/fastflow_sweep/$tag.done
done < "$1"
