#!/bin/bash

set -x
set -e

for filepath in $(ls data/ls_ll/ls_ll_7k_subsets/* | shuf); do
    if [[ $filepath == *done ]]; then
        continue
    fi

    if python3 scripts/utils/check_pt_valid.py $filepath feat/ls_ll_7k_vae_25hz_latent8/; then
        continue
    fi
    python3 scripts/utils/extract_vae_moment.py \
        /home/leo1994122701/cslm/cst/result/vae/vae_lr5e-4_spin_stack2_4transformer_latent8_kl1e-4/best_valid_loss--1.036.ckpt \
        $filepath feat/ls_ll_7k_vae_25hz_latent8/
done
