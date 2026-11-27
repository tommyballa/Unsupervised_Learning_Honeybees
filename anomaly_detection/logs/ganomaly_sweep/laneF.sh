#!/bin/bash
R="$(dirname "$0")/run.sh"
while pgrep -f "sweep_ganomaly.py --tag d1_" >/dev/null; do sleep 10; done
$R f0_p128_512 --aug --img-size 512 --patch 128 --train-images 2000 --epochs 15
$R f1_p32_256 --aug --img-size 256 --patch 32 --epochs 15
