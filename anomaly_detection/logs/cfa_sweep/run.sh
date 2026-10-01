#!/bin/bash
# uso: logs/cfa_sweep/run.sh <tag> [argomenti di sweep_cfa.py]  -> finestra cfa_<tag> della sessione tmux 9, GPU 2
tag=$1; shift
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
tmux new-window -d -t 9: -n "cfa_$tag" "CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/single_bee/sweep_cfa.py --tag $tag $* > logs/cfa_sweep/$tag.log 2>&1"
