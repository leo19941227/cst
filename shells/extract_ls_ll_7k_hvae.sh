#!/bin/bash

set -x
set -e

for filepath in $(ls data/ls_ll/ls_ll_7k_subsets/* | shuf);
do
    if [[ $filepath == *done ]]; then
        continue
    fi
    done_file=${filepath}.done
    if [ -f $done_file ]; then
        continue
    fi
    python3 scripts/utils/extract_hierachy_latent.py \
        result/vae/ae_lr5e-4_1transformer/best_valid_loss--2.156.ckpt \
        result/temporal/ae_lr5e-4_spatial384_stack4_4transformer_latent64/best_valid_loss-0.419.ckpt \
        4 $filepath feat/ls_ll_7k_hvae/
    touch $done_file
done
