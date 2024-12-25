# prune_ratio=0.7
# thr=0.05
# pruner = [reinit / ours]

pruning_ratio=0.9
threshold=0.00
filename=$(basename "$1")

python finetune_simple.py \
--config celeba.yml \
--exp run/sample/celeba_ours_prune_ratio=$pruning_ratio\_T=$threshold.pth/$filename \
--doc sample_50k \
--sample \
--fid \
--eta 0 \
--timesteps 100 \
--skip_type uniform \
--ni \
--use_ema \
--restore_from "$1"