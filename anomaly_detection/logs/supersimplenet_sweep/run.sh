#!/bin/bash
# usage: logs/supersimplenet_sweep/run.sh <gpu> <tag> [sweep_supersimplenet.py arguments]  -> window ssn_<tag> of tmux session 13
gpu=$1; tag=$2; shift 2
cd "$(dirname "$0")/../.."   # anomaly_detection/
tmux new-window -d -t 13: -n "ssn_$tag" "CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/single_bee/sweep_supersimplenet.py --tag $tag $* > logs/supersimplenet_sweep/$tag.log 2>&1"
