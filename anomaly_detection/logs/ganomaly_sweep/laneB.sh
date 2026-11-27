#!/bin/bash
R="$(dirname "$0")/run.sh"
$R b0_aug256 --aug --img-size 256
$R b1_aug256_lat512 --aug --img-size 256 --latent 512
