# prune_ratio=0.3
# thr=0.05
# pruner = [reinit / ours]

pruning_ratio=0.3
threshold=0.05

python finetune_simple.py \
--config cifar10.yml \
--timesteps 100 \
--eta 0 \
--ni \
--exp run/prune_ckpt/cifar10_ours_prune_ratio=$pruning_ratio\_T=$threshold.pth \
--doc post_training \
--skip_type quad  \
--pruning_ratio $pruning_ratio \
--use_ema \
--use_pretrained \
--thr $threshold \
--pruner ours \
--prune_only \

