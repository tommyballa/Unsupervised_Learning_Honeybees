#!/bin/bash
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
export CUDA_VISIBLE_DEVICES=2
.venv/bin/python experiments/single_bee/main_ganomaly.py > logs/ganomaly_main.log 2>&1 \
  && .venv/bin/python experiments/plot_heatmaps.py single_bee ganomaly > logs/ganomaly_plot.log 2>&1
echo "exit $?"
