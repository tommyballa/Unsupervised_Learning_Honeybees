#!/bin/bash
# uso: logs/supersimplenet_sweep/run.sh <gpu> <tag> [argomenti di sweep_supersimplenet.py]  -> finestra ssn_<tag> della sessione tmux 13
gpu=$1; tag=$2; shift 2
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
tmux new-window -d -t 13: -n "ssn_$tag" "CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/single_bee/sweep_supersimplenet.py --tag $tag $* > logs/supersimplenet_sweep/$tag.log 2>&1"
