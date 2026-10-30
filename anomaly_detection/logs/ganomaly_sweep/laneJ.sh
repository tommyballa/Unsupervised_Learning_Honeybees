#!/bin/bash
R=/home/tommaso_ballarin/moviad_acquario/tesi_v2/logs/ganomaly_sweep/run.sh
B="--aug --img-size 256 --patch 32 --ema 0.999 --cosine"
$R j0_ema_cos_e80 $B --epochs 80 --eval-every 10
$R j1_ema_cos_lat256 $B --epochs 40 --latent 256
