# prune_ratio=0.5
# thr=0.00
# pruner = [reinit / ours]

pruning_ratio=0.5
threshold=0.00

python finetune_simple.py \
--config celeba.yml \
--timesteps 100 \
--eta 0 \
--ni \
--exp run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio\_T=$threshold.pth \
--doc post_training \
--skip_type uniform  \
--pruning_ratio $pruning_ratio \
--use_ema \
--use_pretrained \
--thr $threshold \
--pruner ours \
--load_pruned_model run/prune_ckpt/celeba_ours_prune_ratio=$pruning_ratio\_T=$threshold.pth/logs/post_training/pruned_model.pth \