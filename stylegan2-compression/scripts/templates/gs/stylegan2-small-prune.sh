#!/bin/bash

python prune_gs.py --outdir checkpoints/gs/pruned \
                --dataset  \
                --cfg paper256 \
                --gpus 1 \
                --dense  \
                --prune-ratio 0.7 \
                --n-sample 400 \
                --batch 8 \

# python prune_gs.py --outdir checkpoints/gs/pruned \
#                 --dataset ffhq256x256 \
#                 --cfg paper256 \
#                 --gpus 1 \
#                 --dense  \
#                 --prune-ratio 0.7 \
#                 --n-sample 400 \
#                 --batch 8 \
