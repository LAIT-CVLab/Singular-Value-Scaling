python finetune.py \
--config celeba.yml \
--exp run/sample/ddim_celeba_official_1gpu \
--doc sample_50k \
--sample \
--fid \
--eta 0 \
--timesteps 100 \
--skip_type uniform  \
--use_pretrained \
--ni \
--pruning_ratio 0.0 \
--use_ema \