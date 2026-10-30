#!/bin/bash
R=/home/tommaso_ballarin/moviad_acquario/tesi_v2/logs/ganomaly_sweep/run.sh
B="--aug --img-size 256 --patch 32 --ema 0.999 --cosine"
while pgrep -f "sweep_ganomaly.py --tag h1_" >/dev/null; do sleep 10; done
$R j2_ema_cos_noadv $B --epochs 40 --wadv 0
$R j3_ema_cos_lat50 $B --epochs 40 --latent 50
