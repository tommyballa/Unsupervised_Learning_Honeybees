#!/bin/bash
# uso: queue.sh <file con una run per riga: tag args...> (rilegge il file dopo ogni run)
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
while true; do
  line=$(while read -r tag args; do [ -z "$tag" ] && continue; [[ "$tag" == \#* ]] && continue; [ -f logs/paste_sweep/$tag.done ] || [ -f logs/paste_sweep/$tag.running ] || { echo "$tag $args"; break; }; done < "$1")
  [ -z "$line" ] && break
  read -r tag args <<< "$line"
  touch logs/paste_sweep/$tag.running
  CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -u experiments/single_bee/sweep_paste.py --tag $tag $args > logs/paste_sweep/$tag.log 2>&1 && touch logs/paste_sweep/$tag.done
  rm -f logs/paste_sweep/$tag.running
done
