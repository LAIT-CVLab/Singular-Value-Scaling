# StyleGAN3 Compression

- This repository is built upon [DCP-GAN](https://github.com/jiwoogit/DCP-GAN/tree/stylegan3)

## Usage

### Installation

- We use following conda environments for all experiments.

```
conda create -n stylegan2-dev -y python=3.8
conda activate stylegan3-dev
conda install -y cudatoolkit=11.1 -c nvidia
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y libxcrypt gxx_linux-64=7 cxx-compiler ninja cudatoolkit-dev -c conda-forge
pip install click requests tqdm opencv-python-headless matplotlib regex ftfy psutil scipy numba ipython scikit-image==0.15.0 scikit-learn==0.23.2
```

### Prepare datsets

- You can prepare dataset following [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file#preparing-datasets) respository.

### Download Dense StyleGAN2 Weights

#### 1. Pre-trained compresesed weights for inference download: [here](https://drive.google.com/drive/folders/1uNKHBu4t77l2jHtCCxkRfQmJ2QMbvDDh?usp=drive_link).

#### 2. Pre-trained dense weights for pruning & fine-tuning:
- FFHQ-256 : [DCP-GAN](https://github.com/jiwoogit/DCP-GAN?tab=readme-ov-file#pre-trained-weights) github repostiory. (`StyleGAN3_FFHQ256` dir)

### Prune Models

- We prune the pre-trained weights using DCP-GAN method.

Copy the teacher weight 'stylegan3-ffhq-256x256.pkl' to the 'weights' directory, then run:

```
python prune_diversity.py \
    --network weights/stylegan3-ffhq-256x256.pkl \
    --outdir weights/pruned_weights
```

- **Pruning ratio ($p_r$)** is controlled by the `--pruning_ratio` parameter. (default: 0.7)
- **Strength of perturbation ($\alpha$)** is controlled by the `--edit_strength` parameter. (default: 5.0)
- **The number of perturbations for each latent vector ($N$)** is controlled through the `--n_direction` parameter. (default: 10)

### Finetune Models

```
python train_scaling.py \
    --outdir /path/to/save/training/results \
    --cfg stylegan3-t \
    --data /path/to/dataset \
    --gpus 4 \
    --batch 16 \
    --batch-gpu 4 \
    --mirror 1 \
    --gamma 10 \
    --mirror 0 \
    --kimg 15000 \
    --cbase 16384 \
    --teacher_pkl /path/to/pretrained/weights \
    --pruned_pth /path/to/pruned/weights \
    --pruning_ratio 0.7 \
    --stylekd 0 \
    --load-torgb 1 \
    --scaling 1 \
```

## Evaluations


### FID

```
python calc_metrics.py --metrics=fid50k_full --data=/path/to/dataset --mirror=1 --network=/path/to/compressed/weights
```

### Precision & Recall & Density & Coverage

```
# For using full real samples & 50K fake samples
python calc_metrics.py --metrics=prdc50k_full --data=/path/to/dataset --mirror=1 --network=/path/to/compressed/weights
```

## Generate Samples

```
python generate.py --outdir=/path/to/save/samples --trunc=1.0 --seeds=30-42 --network=/path/to/compressed/weights
```

## Citations

```
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

This repository heavily depends on [StyleGAN3](https://github.com/NVlabs/stylegan3), [DCP-GAN](https://github.com/jiwoogit/DCP-GAN)