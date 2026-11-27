#!/bin/bash
R="$(dirname "$0")/run.sh"
B="--aug --img-size 256 --patch 32 --epochs 40"
$R i1_p32_256_ema $B --ema 0.999
$R i0_p32_256_noadv $B --wadv 0
$R i2_p32_256_ema_cos $B --ema 0.999 --cosine
