pruning_ratio=0.3
threshold=0.00

for step in {20000..200000..20000}
do
    bash scripts/exp/celeba_our_sample_fid_pr${pruning_ratio}_thr${threshold}.sh run/finetune_simple/celeba_ours_prune_ratio=${pruning_ratio}_T=${threshold}.pth/logs/post_training/ckpt_${step}.pth
done

# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_20000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_40000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_60000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_80000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_100000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_120000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_140000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_160000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_180000.pth
# bash scripts/exp/celeba_our_sample_fid_pr$pruning_ratio_thr$threshold.sh run/finetune_simple/celeba_ours_prune_ratio=$pruning_ratio_T=$threshold.pth/logs/post_training/ckpt_200000.pth
# 
