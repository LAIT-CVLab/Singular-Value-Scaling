#!/bin/bash

python train.py --outdir checkpoints/train \
                --data /path/to/dataset \
                --mirror 1 \
                --cfg stylegan2 \
                --aug noaug \
                --gpus 4 \

# python train.py --outdir checkpoints/train \
#                 --data /path/to/dataset \
#                 --mirror 1 \
#                 --cfg stylegan2 \
#                 --aug noaug \
#                 --gpus 4 \
#                 --resume-train /path/to/last_pkl \
