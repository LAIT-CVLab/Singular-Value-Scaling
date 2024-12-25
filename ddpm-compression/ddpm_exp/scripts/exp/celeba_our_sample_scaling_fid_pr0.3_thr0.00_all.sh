pruning_ratio=0.3
threshold=0.00

for step in {20000..200000..20000}
do
    bash scripts/exp/celeba_our_sample_scaling_fid_pr${pruning_ratio}_thr${threshold}.sh run/finetune_simple_scaling/celeba_ours_prune_ratio=${pruning_ratio}_T=${threshold}.pth/logs/post_training/ckpt_${step}.pth
done
