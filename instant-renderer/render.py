import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_ortho_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

import datasets
import systems
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils.callbacks import CodeSnapshotCallback, ConfigSnapshotCallback, CustomProgressBar
from utils.misc import load_config

import argparse
import os
from datetime import datetime

import pdb

def get_ortho_rays(origins, directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3
    assert origins.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = torch.matmul(c2w[:, :3, :3], directions[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = torch.matmul(c2w[:, :3, :3], origins[:, :, None]).squeeze()  # (N_rays, 3)
        rays_o = c2w[:,:3,3].expand(rays_d.shape) + rays_o  
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = torch.matmul(c2w[None, None, :3, :3], directions[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = torch.matmul(c2w[None, None, :3, :3], origins[:, :, :, None]).squeeze()  # (H, W, 3)
            rays_o = c2w[None, None,:3,3].expand(rays_d.shape) + rays_o  
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = torch.matmul(c2w[:,None, None, :3, :3], directions[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = torch.matmul(c2w[:,None, None, :3, :3], origins[None, :, :, :, None]).squeeze()  # # (B, H, W, 3)
            rays_o = c2w[:,None, None, :3,3].expand(rays_d.shape) + rays_o  

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default='exp/owl/@20231225-065200/ckpt/epoch=0-step=10000.ckpt', help='path to the weights to be resumed')
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--validate', action='store_true')
    group.add_argument('--test', action='store_true')
    group.add_argument('--predict', action='store_true')
    # group.add_argument('--export', action='store_true') # TODO: a separate export action

    parser.add_argument('--exp_dir', default='./exp')
    parser.add_argument('--runs_dir', default='./runs')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.trial_name = config.get('trial_name') or 'test_renderer' # (config.tag + datetime.now().strftime('@%Y%m%d-%H%M%S'))
    config.exp_dir = config.get('exp_dir') or os.path.join(args.exp_dir, config.name)
    config.save_dir = config.get('save_dir') or os.path.join(config.exp_dir, config.trial_name, 'save')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')
    config.config_dir = config.get('config_dir') or os.path.join(config.exp_dir, config.trial_name, 'config')

    if 'seed' not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None if not args.resume_weights_only else args.resume)

    strategy = 'ddp_find_unused_parameters_false'
    trainer = Trainer(
        devices=n_gpus,
        accelerator='gpu',
        callbacks=None,
        logger=None,
        strategy=strategy,
        **config.trainer
    )
    trainer.validate(system, datamodule=dm, ckpt_path=args.resume) # render multi-view images
    # trainer.test(system, datamodule=dm, ckpt_path=args.resume) # extract mesh
    # pdb.set_trace()

if __name__ == '__main__':
    main()