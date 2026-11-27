#!/bin/bash
R="$(dirname "$0")/run.sh"
while pgrep -f "sweep_ganomaly.py --tag a2_" >/dev/null; do sleep 10; done
$R c0_p64_256 --aug --img-size 256 --patch 64
$R c1_p32_256 --aug --img-size 256 --patch 32
$R c2_p64_256_lat16 --aug --img-size 256 --patch 64 --latent 16
