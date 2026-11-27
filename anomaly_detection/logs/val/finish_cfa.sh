#!/bin/bash
# Coda GPU1 divisa: aspetta la fine di main_cfa.py (PID passato) e ne fa il plot.
cd "$(dirname "$0")/../.."   # anomaly_detection/
while kill -0 "$1" 2>/dev/null; do sleep 30; done
if tr '\r' '\n' < logs/val/cfa_main.log | grep -aq '^Best CFA'; then
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/plot_heatmaps.py single_bee cfa > logs/val/cfa_plot.log 2>&1 \
    && touch logs/val/cfa.done || touch logs/val/cfa.failed
else
  touch logs/val/cfa.failed
fi
echo "$(date '+%F %T') GPU1 end cfa ($(ls logs/val/cfa.done logs/val/cfa.failed 2>/dev/null | xargs -n1 basename))" >> logs/val/progress.txt
