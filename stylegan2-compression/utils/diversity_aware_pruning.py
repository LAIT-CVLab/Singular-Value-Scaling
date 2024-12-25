import torch
from torch import nn
from torchvision import utils, transforms
import torch.nn.functional as F
from torch_utils import misc

import numpy as np
from PIL import Image

from .estimator import get_estimator
from pathlib import Path
from tqdm import tqdm
file_path = Path(__file__).parent

def vis_parsing_maps(im, parsing_anno, stride):
    import cv2
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im


def Get_Weight_Gradient(noisy_img, img_tensor, generator):
    '''
    Usage:
        Obtain the gradients of all filters' weights in the feed-forward path
    
    Args:
        noisy_img:  (torch.Tensor) of the noisy image
        img_tensor: (torch.Tensor) of the original generated image
        generator:  (nn.Module) of the generator
    '''
    loss = torch.mean(torch.abs(noisy_img - img_tensor))
    loss.backward()
    
    resolution = generator.img_resolution
    module_list = []
    for n, m in generator.synthesis.named_modules():
        if hasattr(m, "conv0"):
            module_list.append(m.conv0)
        if hasattr(m, "conv1"):
            module_list.append(m.conv1)
            
    module_list.append(getattr(generator.synthesis, f"b{resolution}").torgb)
    grad_list = [module.weight.grad for module in module_list]
    # grad_list = [misc.nan_to_num(grad) for grad in grad_list]

    # grad_score_list = [misc.nan_to_num((torch.mean(torch.abs(grad), axis=[0,2,3]))).cpu().numpy() for grad in grad_list]
    grad_score_list = [(torch.mean(torch.abs(grad), axis=[0,2,3])).cpu().numpy() for grad in grad_list]
    # grad_score_list = [(torch.mean(grad, axis=[0,2,3])).cpu().numpy() for grad in grad_list]

    return grad_score_list

def get_diversity_pruning_score(g, n_sample, batch_size, device, \
    edit_strength, n_direction, noise_path=None, info_print=False):
    '''
    Usage:
        Obtain the network score
    
    Args:
        g:             (Module) of a generator
        n_sample:      (int) of the number of samples for estimation
        batch_size:    (int) of the size of the batch
        device:        (str) the device to place for the operations
        edit_strength: (float) of the strength of the perturbations
        n_direction:   (int) of the number of perturbation latent vectors.
        noise_path:    (str) the path of the z (reproduce result)
    '''
    # noise and batch setup
    LATENT_DIM = 512 
    n_components = n_direction
    alpha = edit_strength
    n_batch = n_sample // batch_size
    batch_size_list = [batch_size] * (n_batch - 1) + [batch_size + n_sample % batch_size]
    grad_score_list = []
    transformer = get_estimator('pca', LATENT_DIM, None)


    noise_z = torch.randn(10000, LATENT_DIM).to(device)
    latents = g.mapping(noise_z, None)[:, 0, :]
    transformer.fit(latents.detach().cpu().numpy())
    comp, stddev, var_ratio = transformer.get_components()
    comp /= np.linalg.norm(comp, axis=1, keepdims=True)
    comp = torch.from_numpy(comp).to(device)
    num_ws = g.synthesis.num_ws
    print('total var sum: ',  sum(var_ratio))
    if noise_path is not None:
        noise_z_load = torch.load(noise_path).to(device)
    
    for (idx,batch) in enumerate(tqdm(batch_size_list)):
        if info_print:
            print('Processing Batch: ' + str(idx))
        noise_z = torch.randn(batch, LATENT_DIM).to(device)
        if noise_path is not None:
            noise_z = noise_z_load[idx*batch_size:idx*batch_size+batch]

        grad_score = []
        comp_list = np.random.choice(512, n_components, p=var_ratio)
        comp_list = comp[comp_list]

        for i in range(n_components):
            latents = g.mapping(noise_z, None)[:, 0, :]
            direction = comp_list[i].unsqueeze(0).repeat(batch, 1)
            latents_pca = latents.clone().detach() + (direction * alpha)
            input_ws_pca = latents_pca.unsqueeze(1).repeat(1, num_ws, 1)
            input_ws = latents.unsqueeze(1).repeat(1, num_ws, 1)

            # stop grad
            g.requires_grad_(False)
            img_pca = g.synthesis(input_ws_pca)
            g.requires_grad_(True)
            img = g.synthesis(input_ws)

            grad_score_1 = Get_Weight_Gradient(img_pca, img, g)
            g.zero_grad()

            grad_score.append(grad_score_1)

        n_layer = len(grad_score[0])
        mean_grad = []
        for n in range(n_layer):
            all_other_grad = []
            for i in range(len(grad_score)):
                all_other_grad.append(grad_score[i][n])
            all_other_grad = np.stack(all_other_grad, axis=0).mean(axis=0)
            mean_grad.append(all_other_grad)

        for i in range(len(grad_score)):
            tmp = []
            for n in range(len(grad_score[0])):
                
                # print(grad_score[i][n].mean(), mean_grad[n].mean(), ((grad_score[i][n] - mean_grad[n])**2).mean())
                
                tmp.append((grad_score[i][n] - mean_grad[n])**2)
                # tmp.append(np.abs(grad_score[i][n] - mean_grad[n]))
            # print(tmp)
            grad_score_list.append(tmp)
        
    return grad_score_list