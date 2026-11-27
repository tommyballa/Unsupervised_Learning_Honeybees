#!/bin/bash
R="$(dirname "$0")/run.sh"
while pgrep -f "sweep_ganomaly.py --tag e1_" >/dev/null; do sleep 10; done
$R h0_p32_512 --aug --img-size 512 --patch 32 --train-images 2000 --epochs 20
$R h1_p32_128 --aug --img-size 128 --patch 32 --epochs 40
