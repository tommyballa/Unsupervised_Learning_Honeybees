#!/bin/bash
R=/home/tommaso_ballarin/moviad_acquario/tesi_v2/logs/ganomaly_sweep/run.sh
$R a0_base128
$R a1_aug128 --aug
$R a2_aug128_lat512 --aug --latent 512
$R a3_aug128_l2 --aug --con-loss l2
