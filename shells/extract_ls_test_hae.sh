#!/bin/bash

set -x
set -e

python3 scripts/utils/extract_hierachy_latent.py \
    result/vae/ae_lr5e-4_1transformer/best_valid_loss--2.156.ckpt \
    result/temporal/ae_lr5e-4_spatial384_stack4_4transformer_latent64/best_valid_loss-0.419.ckpt \
    4 data/librispeech/test_clean feat/ls_test_hvae/

