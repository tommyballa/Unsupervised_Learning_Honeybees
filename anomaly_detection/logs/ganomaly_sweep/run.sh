#!/bin/bash
# uso: run.sh <tag> [argomenti di sweep_ganomaly.py]
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
tag=$1; shift
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/single_bee/sweep_ganomaly.py --tag "$tag" "$@" > "logs/ganomaly_sweep/$tag.log" 2>&1
