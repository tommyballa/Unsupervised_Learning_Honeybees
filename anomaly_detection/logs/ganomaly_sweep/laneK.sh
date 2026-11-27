#!/bin/bash
R="$(dirname "$0")/run.sh"
B="--aug --img-size 256 --patch 32 --ema 0.999 --cosine"
while pgrep -f "sweep_ganomaly.py --tag h1_" >/dev/null; do sleep 10; done
$R j2_ema_cos_noadv $B --epochs 40 --wadv 0
$R j3_ema_cos_lat50 $B --epochs 40 --latent 50
