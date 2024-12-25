pruning_ratio=0.5
threshold=0.00

# pretrained_model path: https://github.com/ermongroup/ddim/tree/main?tab=readme-ov-file"

python finetune_simple.py \
--config celeba.yml \
--timesteps 100 \
--eta 0 \
--ni \
--exp run/prune_ckpt/celeba_ours_prune_ratio=$pruning_ratio\_T=$threshold.pth \
--doc post_training \
--skip_type uniform  \
--pruning_ratio $pruning_ratio \
--use_ema \
--use_pretrained \
--thr $threshold \
--pruner ours \
--taylor_batch_size 64 \
--prune_only \