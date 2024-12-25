import os
import click
import numpy as np
import copy
import json
import pickle
import torch
import dnnlib
import legacy
from torch_utils import misc

from utils.mask_util import mask_the_generator
from utils.utils import set_random_seed
from utils.pruning_util import get_pruning_scores


def setup_training_loop_kwargs(
    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    prune_ratio= 0.0,  # Student Generator Pruning Ratio: <float>, default = 0.0
    n_sample   = 400,   # Total number of samples for estimation: <int>, default = 400
    batch      = 8,    # Size of the batch for estimation: <int>, default = 8
):
    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    args = dnnlib.EasyDict()
        
    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])

    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
        
    args.prune_criterion = 'GS'
    
    if prune_ratio is None:
        prune_ratio = 0.0
    assert isinstance(prune_ratio, float) and 0.0 <= prune_ratio <= 1.0
    args.prune_ratio = prune_ratio
        
    if n_sample is None:
        n_sample = 400
    assert isinstance(n_sample, int)
    args.n_sample = n_sample
    
    if batch is None:
        batch = 8
    assert isinstance(batch, int)
    args.batch_size = batch    
    
    # Set Network Pruning Ratio
    args.G_kwargs["class_name"] = "training.networks_gs.Generator"
    args.G_teacher_kwargs = args.G_kwargs.copy()
    args.G_student_kwargs = args.G_kwargs.copy()
    args.G_student_kwargs["prune_ratio"] = args.prune_ratio
    del args.G_kwargs
    
    return args

@click.command()

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']))
@click.option('--dataset', help='dataset dense model trained with', type=str, required=True, metavar='DATASET')

# GAN Compression pruning options (not included in desc).
@click.option('--dense', help='Dense Generator network pickle', type=str, metavar='PKL')
@click.option('--prune-ratio', help='Student Generator Pruning Ratio', type=float, metavar='FLOAT')
@click.option('--n-sample', help='Size of the batch for estimation', type=int, metavar='INT')
@click.option('--batch', help='Size of the batch for estimation', type=int, metavar='INT')

def main(outdir, dataset, gpus, dense, **config_kwarg):
    set_random_seed(0)
    assert isinstance(dataset, str)
    
    if gpus is None:
        gpus = 0
    assert isinstance(gpus, int)
    if gpus > 0:
        gpus = 1
    
    device = torch.device('cuda') if gpus > 0 else torch.device('cpu')
    
    assert isinstance(dense, str) and os.path.isfile(dense)
    
    args = setup_training_loop_kwargs(**config_kwarg)
    args.Dense_model_path = dense
    
    print(f'full model loading from "{dense}"')
    with dnnlib.util.open_url(dense) as f:
        load_data = legacy.load_network_pkl(f)
    
    common_kwargs = dict(c_dim=load_data['G_ema'].c_dim, img_resolution=load_data['G_ema'].img_resolution, img_channels=load_data['G_ema'].img_channels)
    
    G_ema = dnnlib.util.construct_class_by_name(**args.G_teacher_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**args.D_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    for name, module in [('D', D), ('G_ema', G_ema)]:
        if module is not None:
            misc.copy_params_and_buffers(load_data[name], module, require_all=False) # dense model load
            
    score_list = get_pruning_scores(generator = G_ema,
                                    discriminator = D, 
                                    args = args,
                                    device = device)
    score_array = np.array([np.array(score) for score in score_list])    
    pruning_score = np.sum(score_array, axis=0)   
    
    pruned_generator_dict = mask_the_generator(G_ema.state_dict(), pruning_score, args)
    
    G_pruned = dnnlib.util.construct_class_by_name(**args.G_student_kwargs, **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_pruned.load_state_dict(pruned_generator_dict)
    
    for name, module in [('G', G_pruned), ('D', D), ('G_ema', G_pruned)]:
        if module is not None:
            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        load_data[name] = module
        del module # conserve memory
        
    
    # Save path settings    
    save_path = os.path.join(outdir, f'pruning-ratio-{args.prune_ratio}', dataset)
     
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'pruning_options.json'), 'wt') as f:
        json.dump(args, f, indent=4)    
    snapshot_pkl = os.path.join(save_path, f'pruned-network-{args.prune_ratio}.pkl')

    with open(snapshot_pkl, 'wb') as f:
        pickle.dump(load_data, f)
        
    print()
    print('Exiting...')
    
if __name__== "__main__":
    main()