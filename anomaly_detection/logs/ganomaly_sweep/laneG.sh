#!/bin/bash
R=/home/tommaso_ballarin/moviad_acquario/tesi_v2/logs/ganomaly_sweep/run.sh
$R g0_p32_256_e40 --aug --img-size 256 --patch 32 --epochs 40
$R g2_p32_256_lat32 --aug --img-size 256 --patch 32 --epochs 40 --latent 32
