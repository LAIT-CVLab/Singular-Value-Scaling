#!/bin/bash

python train_gs.py --outdir checkpoints/gs/finetune \
                --data /path/to/dataset \
                --mirror 1 \
                --cfg stylegan2 \
                --aug noaug \
                --gpus 4 \
                --teacher  \
                --student  \
                --prune-ratio 0.7 \
                --kd-lambda 3.0 \

# python train_gs.py --outdir checkpoints/gs/finetune \
#                 --data /path/to/dataset \
#                 --mirror 1 \
#                 --cfg stylegan2 \
#                 --aug noaug \
#                 --gpus 4 \
#                 --resume-train /path/to/last_pkl \
#                 --teacher  \
#                 --student  \
#                 --prune-ratio 0.7 \
#                 --kd-lambda 3.0 \

