#!/bin/bash
R="$(dirname "$0")/run.sh"
while pgrep -f "sweep_ganomaly.py --tag b0_" >/dev/null; do sleep 10; done
$R d0_p64_512 --aug --img-size 512 --patch 64 --train-images 2000
$R d1_p64_512_lat16 --aug --img-size 512 --patch 64 --train-images 2000 --latent 16
