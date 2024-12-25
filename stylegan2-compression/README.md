# StyleGAN2 Compression

## Usage

### Installation

- We used following conda environments for all experiments.
```
conda create -n stylegan2-dev -y python=3.8
conda activate stylegan2-dev
conda install -y cudatoolkit=11.1 -c nvidia
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y libxcrypt gxx_linux-64=7 cxx-compiler ninja cudatoolkit-dev -c conda-forge
pip install click requests tqdm opencv-python-headless matplotlib regex ftfy psutil scipy numba ipython scikit-image==0.15.0 scikit-learn==0.23.2 tensorboard
```

### Prepare datsets

- You can prepare dataset following [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file#preparing-datasets) respository.

### Download Dense StyleGAN2 Weights

#### 1. Pre-trained compresesed weights for inference download: [here](https://drive.google.com/drive/u/1/home).

#### 2-1. Pre-trained dense weights for pruning & fine-tuning:
- FFHQ-256 / LSUN Church-256 (base) : [DCP-GAN](https://github.com/jiwoogit/DCP-GAN?tab=readme-ov-file#pre-trained-weights) github repostiory.
- FFHQ-256 (small) : [Nvidia-official FFHQ-256 weights](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl).
- FFHQ-1024 : [Nvidia-official FFHQ-1024 weights](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl).
- For remaining datsets, you can download the weights from [here](https://drive.google.com/drive/u/1/home).

#### 2-2. Weight convert for unofficial implementation.
- FFHQ-256 / LSUN Church-256 (base) models provided by [DCP-GAN](https://github.com/jiwoogit/DCP-GAN?tab=readme-ov-file#pre-trained-weights) are trained with an unofficial implementation. You need to convert the weights to be compatible with the official implementation. You can use "weight_converter.ipynb".

### Prune Models

- We prune the pre-trained weights using DCP-GAN method.
    ```
    python prune_dcp.py --outdir /path/to/save/pruned/weights \
                --dataset /dataset/description \
                --cfg paper256 \
                --gpus 1 \
                --dense /path/to/pretrained/weights \
                --prune-ratio 0.7 \
                --n-sample 5000 \
                --batch 5 \
                --edit-strength 10.0 \
                --n-direction 10
    ```

### Finetune Models
```
python train_stylekd_scaling.py --outdir /path/to/save/training/results \
                --data /path/to/dataset \
                --mirror 1 \
                --cfg paper256 \ [paper256 | stylegan2]
                --aug noaug \
                --gpus 4 \
                --teacher /path/to/pretrained/weights \
                --student /path/to/pruned/weights \
                --prune-ratio 0.7 \
                --kd-l1-lambda 3.0 \
                --kd-lpips-lambda 3.0 \
                --kd-simi-lambda 30.0 \
                --pretrained-discriminator 0 \
                --load-torgb 1 \
                --load-style 0 \
                --initialize scaling \
                --initialize-torgb scaling \
                --mimic-layer 2,3,4,5 \
                --simi-loss kl \
                --single-view 0 \
                --offset-mode main \
                --offset-weight 5.0 \
                --main-direction split \
                --dataset ffhq \ [ffhq | church]
```

- We provide example scripts in [scripts](scripts/) directory.

## Evaluations

### Model FLOPs

```
python calculate_flops.py --network /path/to/compressed/weights
```

### FID
```
python calc_metrics.py --metrics=fid50k_full --data=/path/to/dataset --mirror=1 --network=/path/to/compressed/weights
```
### Precision & Recall & Density & Coverage
```
# For using full real samples & 50K fake samples
python calc_metrics.py --metrics=prdc50k_full --data=/path/to/dataset --mirror=1 --network=/path/to/compressed/weights
# For using 50K real samples & 50K fake samples
python calc_metrics.py --metrics=prdc50k --data=/path/to/dataset --mirror=1 --network=/path/to/compressed/weights
```
## Generate Samples
```
python generate.py --outdir=/path/to/save/samples --trunc=1.0 --seeds=30-42 --network=/path/to/compressed/weights
```
## Citations

```
@article{ferjad2020icml,
  title = {Reliable Fidelity and Diversity Metrics for Generative Models},
  author = {Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  year = {2020},
  booktitle = {International Conference on Machine Learning},
}
@inproceedings{Karras2020ada,
    title     = {Training Generative Adversarial Networks with Limited Data},
    author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
    booktitle = {Proc. NeurIPS},
    year      = {2020}
}
@inproceedings{liu2021content,
    title     = {Content-Aware GAN Compression},
    author    = {Liu, Yuchen and Shu, Zhixin and Li, Yijun and Lin, Zhe and Perazzi, Federico and Kung, S.Y.},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
}
@misc{xu2022stylekd,
  url = {https://arxiv.org/abs/2208.08840},
  author = {Xu, Guodong and Hou, Yuenan and Liu, Ziwei and Loy, Chen Change},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Mind the Gap in Distilling StyleGANs},
  publisher = {arXiv},
  year = {2022}
}
@InProceedings{Chung_2024_CVPR,
    author    = {Chung, Jiwoo and Hyun, Sangeek and Shim, Sang-Heon and Heo, Jae-Pil},
    title     = {Diversity-aware Channel Pruning for StyleGAN Compression},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {7902-7911}
}
```

## Acknowledgement

This repository heavily depends on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), [CAGC](https://github.com/lychenyoko/content-aware-gan-compression), [StyleKD](https://github.com/xuguodong03/StyleKD), [DCP-GAN](https://github.com/jiwoogit/DCP-GAN)