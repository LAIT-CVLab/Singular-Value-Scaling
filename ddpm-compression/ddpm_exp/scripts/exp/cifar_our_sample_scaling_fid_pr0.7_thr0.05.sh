# prune_ratio=0.7
# thr=0.05
# pruner = [reinit / ours]

pruning_ratio=0.7
threshold=0.05
filename=$(basename "$1")

python finetune_simple.py \
--config cifar10.yml \
--exp run/sample/cifar10_ours_scaling_prune_ratio=$pruning_ratio\_T=$threshold.pth/$filename \
--doc sample_50k \
--sample \
--fid \
--eta 0 \
--timesteps 100 \
--skip_type quad  \
--ni \
--use_ema \
--restore_from "$1"