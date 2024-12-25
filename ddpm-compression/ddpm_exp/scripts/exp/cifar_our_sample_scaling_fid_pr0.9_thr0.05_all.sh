pruning_ratio=0.9
threshold=0.05

for step in {40000..400000..40000}
do
    bash scripts/exp/cifar_our_sample_scaling_fid_pr${pruning_ratio}_thr${threshold}.sh run/finetune_simple_scaling/cifar10_ours_prune_ratio=${pruning_ratio}_T=${threshold}.pth/logs/post_training/ckpt_${step}.pth
done
