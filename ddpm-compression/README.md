# Diff-Pruning: Structural Pruning for Diffusion Models

<div align="center">
<img src="assets/framework.png" width="80%"></img>
</div>

### Installation

```
conda create -n diff-prune -y python=3.9
conda activate diff-prune
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y
pip install -r requirements.txt
```

## Our Exp Code (Unorganized)

### Pruning with DDIM codebase
exp code [ddpm_exp](ddpm_exp/scripts/exp/). 

```bash
cd ddpm_exp
# Prune & Finetune
bash scripts/simple_cifar_our.sh 0.05 # the pre-trained model and data will be automatically prepared
# Sampling
bash scripts/sample_cifar_ddpm_pruning.sh run/finetune_simple_v2/cifar10_ours_T=0.05.pth/logs/post_training/ckpt_100000.pth run/sample

```bash
# pre-compute the stats of CIFAR-10 dataset
python fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
```

```bash
# Compute the FID score of sampled images
python fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256

## Citation
If you find this work helpful, please cite:
```
@inproceedings{fang2023structural,
  title={Structural pruning for diffusion models},
  author={Gongfan Fang and Xinyin Ma and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```

```
@inproceedings{fang2023depgraph,
  title={Depgraph: Towards any structural pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16091--16101},
  year={2023}
}
```
