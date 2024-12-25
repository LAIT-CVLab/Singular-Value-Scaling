python prune_dcp.py --outdir /path/to/save/pruned/weights \
            --dataset church256x256 \
            --cfg paper256 \
            --gpus 1 \
            --dense /path/to/pretrained/weights \
            --prune-ratio 0.7 \
            --n-sample 5000 \
            --batch 5 \
            --edit-strength 10.0 \
            --n-direction 10