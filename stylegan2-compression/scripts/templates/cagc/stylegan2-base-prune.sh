#!/bin/bash

python prune_cagc.py --outdir checkpoints/cagc/pruned \
                --dataset ffhq256x256 \
                --cfg stylegan2 \
                --gpus 1 \
                --dense  \
                --prune-ratio 0.7 \
                --n-sample 400 \
                --batch 8 \
                --noise-prob 0.05 \

# python prune_cagc.py --outdir checkpoints/cagc/pruned \
#                 --dataset ffhq256x256 \
#                 --cfg stylegan2 \
#                 --gpus 1 \
#                 --dense  \
#                 --prune-ratio 0.7 \
#                 --n-sample 400 \
#                 --batch 8 \
#                 --noise-prob 0.05 \
