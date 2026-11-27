#!/bin/bash
# Efficiency measurements of all configurations at full N, run sequentially on GPU $GPU
cd "$(dirname "$0")/../../.."   # anomaly_detection/
M="env CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/single_bee/measure_efficiency.py --cpu --cpu-iters 5"
L=logs/ablation/efficiency/run_all.log
for m in patchcore cfa dinomaly padim stfpm fastflow paste rd4ad; do echo "== $m default" >> $L; $M $m >> $L 2>&1; done
for b in resnet18 mobilenet_v2; do echo "== patchcore $b" >> $L; PATCHCORE_BACKBONE=$b $M patchcore >> $L 2>&1; done
for j in cfa:resnet18 cfa:mobilenet_v2 padim:resnet18 padim:mobilenet_v2 stfpm:wide_resnet50_2 stfpm:mobilenet_v2 \
         paste:wide_resnet50_2 paste:mobilenet_v2 fastflow:resnet18 rd4ad:resnet18 dinomaly:vit_small_patch14_reg4_dinov2; do
  IFS=: read -r m b <<< "$j"; echo "== $m $b" >> $L; ABL_BACKBONE=$b $M $m >> $L 2>&1
done
echo ALLDONE >> $L
