#!/bin/bash

python prune_gs.py --outdir checkpoints/gs/pruned \
                --dataset  \
                --cfg stylegan2 \
                --gpus 1 \
                --dense  \
                --prune-ratio 0.7 \
                --n-sample 400 \
                --batch 8 \

# python prune_gs.py --outdir checkpoints/gs/pruned \
#                 --dataset ffhq256x256 \
#                 --cfg stylegan2 \
#                 --gpus 1 \
#                 --dense  \
#                 --prune-ratio 0.7 \
#                 --n-sample 400 \
#                 --batch 8 \
