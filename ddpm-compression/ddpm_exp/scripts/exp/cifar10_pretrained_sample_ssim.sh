python finetune_simple.py \
--config cifar10.yml \
--exp run/sample/ddpm_cifar10_official_ssim \
--doc sample_50k \
--sample \
--fid \
--eta 0 \
--timesteps 100 \
--skip_type quad \
--ni \
--use_ema \
--use_pretrained \