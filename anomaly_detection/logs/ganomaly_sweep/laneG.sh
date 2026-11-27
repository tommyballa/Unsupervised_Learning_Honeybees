#!/bin/bash
R="$(dirname "$0")/run.sh"
$R g0_p32_256_e40 --aug --img-size 256 --patch 32 --epochs 40
$R g2_p32_256_lat32 --aug --img-size 256 --patch 32 --epochs 40 --latent 32
