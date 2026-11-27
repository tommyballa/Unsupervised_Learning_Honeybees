#!/bin/bash
# usage: queue.sh <file with one run per line: tag args...> (re-reads the file after each run)
cd "$(dirname "$0")/../.."   # anomaly_detection/
while true; do
  line=$(while read -r tag args; do [ -z "$tag" ] && continue; [[ "$tag" == \#* ]] && continue; [ -f logs/padim_sweep/$tag.done ] || [ -f logs/padim_sweep/$tag.running ] || { echo "$tag $args"; break; }; done < "$1")
  [ -z "$line" ] && break
  read -r tag args <<< "$line"
  touch logs/padim_sweep/$tag.running
  CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -u experiments/single_bee/sweep_padim.py --tag $tag $args > logs/padim_sweep/$tag.log 2>&1 && touch logs/padim_sweep/$tag.done
  rm -f logs/padim_sweep/$tag.running
done
