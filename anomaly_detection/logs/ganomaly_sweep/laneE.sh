#!/bin/bash
R="$(dirname "$0")/run.sh"
while pgrep -f "sweep_ganomaly.py --tag c2_" >/dev/null; do sleep 10; done
$R e0_p64_512 --aug --img-size 512 --patch 64 --train-images 2000 --epochs 15
$R e1_p64_512_noadv --aug --img-size 512 --patch 64 --train-images 2000 --epochs 15 --wadv 0
