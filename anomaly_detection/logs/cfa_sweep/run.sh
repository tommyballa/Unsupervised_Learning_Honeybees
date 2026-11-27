#!/bin/bash
# usage: logs/cfa_sweep/run.sh <tag> [sweep_cfa.py arguments]  -> window cfa_<tag> of tmux session 9, GPU 2
tag=$1; shift
cd "$(dirname "$0")/../.."   # anomaly_detection/
tmux new-window -d -t 9: -n "cfa_$tag" "CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/single_bee/sweep_cfa.py --tag $tag $* > logs/cfa_sweep/$tag.log 2>&1"
