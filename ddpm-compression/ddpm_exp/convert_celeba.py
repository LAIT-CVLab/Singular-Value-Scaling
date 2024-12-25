import torch

# Downlaod from DDIM official repo
# https://github.com/ermongroup/ddim/tree/main?tab=readme-ov-file
# https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view
celeba64_pretrained_model = 'ckpt.pth'

checkpoint = torch.load(celeba64_pretrained_model, map_location='cpu')[-1]

new_state_dict = {}

for k, v in checkpoint.items():
    name = k.replace('module.', '')  # 'module.'을 제거합니다.
    new_state_dict[name] = v

torch.save(new_state_dict, 'run/cache/diffusion_models_converted/ema_diffusion_celeba_model/model.ckpt')

