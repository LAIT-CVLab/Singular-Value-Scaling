import torch
import numpy as np

# Sampling style per each layer
def get_gan_slimming_scores(generator:torch.nn.Module, args, device):
        
    assert args.n_sample % args.batch_size == 0
    n_batch = args.n_sample // args.batch_size
    batch_size_list = [args.batch_size] * n_batch
    
    style_score_list = []
    
    for (idx, batch) in enumerate(batch_size_list):
        print("Processing Batch: " + str(idx+1))
        z = torch.randn(batch, generator.z_dim, device = device)
        c = torch.empty(batch, generator.c_dim, device = device)
        ws = generator.mapping(z, c)
        _, _styles_list = generator.synthesis(ws, return_style_scalars = True)                  
        _styles_list = [style.detach().cpu() for style in _styles_list]
        _style = torch.mean(torch.abs(torch.cat(_styles_list, dim=1)), axis=0)        
        style_score_list.append(_style)  
    
    return style_score_list
