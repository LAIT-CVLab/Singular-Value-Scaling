import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from tqdm import tqdm
from runners.diffusion_simple import Diffusion
from torchvision import transforms
import torchvision
from datasets import get_dataset, data_transform, inverse_data_transform

from utils import UnlabeledImageFolder

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed")
    parser.add_argument("--taylor_batch_size", type=int, default=128, help="batch size for taylor expansion")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )

    parser.add_argument(
        "--load_pruned_model", type=str, default=None, help="load pruned models"
    )

    parser.add_argument(
        "--save_pruned_model", type=str, default=None, help="load pruned models"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )

    parser.add_argument(
        "--pruner",
        type=str,
        default="taylor",
        choices=["taylor", "random", "magnitude", "reinit", "first_order_taylor", "second_order_taylor", "ours"],
    )

    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="Restore from user a checkpoint",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.0,
        help="pruning ratio",
    )
    
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.pruning_ratio > 0 and args.load_pruned_model is None:
            # Dataset 
            print(config)
            dataset, _ = get_dataset(args, config)
            print(f"Dataset size: {len(dataset)}")
            train_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.taylor_batch_size, shuffle=True, num_workers=4, drop_last=True
            )

            from models.diffusion import AttnBlock
            import torch_pruning as tp
            print("Pruning ...")
            model = runner.model
            model.to(runner.device)
           
            example_inputs = {'x': torch.randn(1, 3, config.data.image_size, config.data.image_size).to(runner.device), 't': torch.ones(1).to(runner.device)}

            if args.pruner == 'taylor':
                imp = tp.importance.TaylorImportance()
            elif args.pruner == 'first_order_taylor':
                imp = tp.importance.FullTaylorImportance(order=1)
            elif args.pruner == 'second_order_taylor':
                imp = tp.importance.FullTaylorImportance(order=2)
            elif args.pruner == 'random' or args.pruner == 'reinit':
                imp = tp.importance.RandomImportance()
            elif args.pruner == 'magnitude':
                imp = tp.importance.MagnitudeImportance()
            elif args.pruner == 'ours':
                imp = tp.importance.TaylorImportance()

            ignored_layers = [model.conv_out]
            channel_groups = {}
            iterative_steps = 1
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=iterative_steps,
                channel_groups =channel_groups,
                ch_sparsity=args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                ignored_layers=ignored_layers,
                root_module_types=[torch.nn.Conv2d, torch.nn.Linear]
            )
            base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

            if 'taylor' in args.pruner or args.pruner=='ours':
                x = next(iter(train_dataloader))
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(runner.device)
                x = data_transform(config, x)
                n = x.size(0)
                e = torch.randn_like(x)
                b = runner.betas
                #t = torch.randint(
                #        low=0, high=runner.num_timesteps, size=(n // 2 + 1,)
                #).to(runner.device)
                #t = torch.cat([t, runner.num_timesteps - t - 1], dim=0)[:n]
                from functions.losses import loss_registry
                
                model.zero_grad()
                max_loss = 0
                for step_k in tqdm(range(1000)):
                    t = torch.ones(n, dtype=torch.long).to(runner.device)*step_k
                    loss = loss_registry[config.model.type](model, x, t, e, b)
                    if args.pruner == 'ours':
                        if loss>max_loss:
                            max_loss = loss
                        if loss<max_loss*args.thr:
                            break
                        #print(loss, max_loss)
                    loss.backward()

            print("============ Before Pruning ============")
            print(model)
            for g in pruner.step(interactive=True):
                g.prune()
            
            if args.pruner == 'reinit':
                def reset_parameters(model):
                    for m in model.modules():
                        if hasattr(m, 'reset_parameters'):
                            m.reset_parameters()
                model.apply(reset_parameters)
            
            macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
            print("============ After Pruning ============")
            print(model)
            print("#Params: {:.4f} M => {:.4f} M".format(base_nparams/1e6, nparams/1e6))
            print("#MACs: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
            del pruner
            # Save pruned model
            print("Saving pruned model as {}".format(os.path.join(args.log_path, "pruned_model.pth")))
            torch.save(
                model,
                os.path.join(args.log_path, "pruned_model.pth"),
            )
            if 'taylor' in args.pruner or args.pruner=='ours':
                print(step_k)
        
        
        if args.load_pruned_model is not None:
            print("Loading pruned model from {}".format(args.load_pruned_model))
            model = torch.load(args.load_pruned_model, map_location='cpu')
            
            print("Reset Group Norm & Scaling weight and bias")
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            
            for module in list(model.modules()):
                if type(module) is torch.nn.GroupNorm:
                    module.apply(reset_parameters)    
            
            for name, _ in list(model.named_parameters()):
                if 'norm' in name:
                    continue
                
                if 'weight' in name:

                    module_name, param_name = name.rsplit('.', 1)
                    module = model
                    for attr in module_name.split('.'):
                        module = getattr(module, attr)
                        
                    W = module.weight.data.numpy()

                    ndim = W.ndim
                    if ndim == 4:
                        co, ci, k, k = W.shape
                        W = W.reshape(co, -1)
                    
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)
                    S_sqrt = np.sqrt(S)
                    W_sqrt = np.dot(U, np.dot(np.diag(S_sqrt), Vt))

                    if ndim==4:
                        W_sqrt = W_sqrt.reshape(co, ci, k, k)
                    module.weight.data = torch.from_numpy(W_sqrt)
                
                if 'bias' in name:
                    module_name, param_name = name.rsplit('.', 1)
                    module = model
                    for attr in module_name.split('.'):
                        module = getattr(module, attr)
                    
                    module.bias.data *= (module.bias.norm() + 1e-8).rsqrt()
            
            save_dir = os.path.dirname(args.load_pruned_model)
            torch.save(
                model,
                os.path.join(save_dir, "pruned_model_scaling.pth"),
            )
            
            exit()
            
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())