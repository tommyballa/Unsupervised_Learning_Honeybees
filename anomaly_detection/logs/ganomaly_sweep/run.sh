#!/bin/bash
# uso: run.sh <tag> [argomenti di sweep_ganomaly.py]
cd "$(dirname "$0")/../.."   # anomaly_detection/
tag=$1; shift
CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/single_bee/sweep_ganomaly.py --tag "$tag" "$@" > "logs/ganomaly_sweep/$tag.log" 2>&1
