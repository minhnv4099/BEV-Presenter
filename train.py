#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import glob
import sys
sys.path.append('.')

import argparse
import os
from os import path as osp

import torch
from mmengine.config import Config, DictAction
from src.runner import Runner
from src.utils.env import find_load_env
from src.utils.logging import getLogger

find_load_env()
logger = getLogger(name="trainer")

DEFAULT_CONFIG = "configs/bevformer_tiny_test.py"
WORK_DIR = "experiment"


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--config',
        help='config file path, locate in ./configs/',
        default=DEFAULT_CONFIG)
    parser.add_argument(
        '--work-dir',
        help='the dir to save logs and checkpoints',
        default=WORK_DIR)
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Whether to resume training. Defaults to False. If '
             '``resume`` is True and ``load_from`` is None, automatically to'
             'find latest checkpoint from ``work_dir``. '
             'If not found, resuming, try resume checkpoint from hub.')
    parser.add_argument(
        '--load_from',
        default=None,
        help='The checkpoint file to load from. Defaults to None.')
    parser.add_argument(
        '--mode',
        default='val',
        choices=['train', 'val', 'predict'],
        help='Train, test or predict. If mode is val, automatic resume the '
             'checkpoint from `load_from`, so provide it.')
    parser.add_argument(
        '--experiment-name',
        default="baseline",
        help="Experiment name, we use it as different configs"
             "Such as v1-resnet-101, the corresponding repo and dir will be created.")
    parser.add_argument(
        '--repo-id',
        default=None,
        help="The repo to save checkpoints during training. "
             "If set organization or username(e.g. 'project', 'username'), the corresponding repo is 'project/experiment`. "
             "If set repo id (e.g. 'project/resnet'), the corresponding repo is 'project/resnet-experiment'.")

    parser.add_argument('--seed', type=int, default=321, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    if args.debug:
        args.config = DEFAULT_CONFIG
    else:
        logger.warning(
            "It's highly recommended to resume config file from experiment dir"
            " to continue training with consistent configs/hyperparameters."
        )

    if not args.config:
        try:
            config_file = f"{args.work_dir}/{args.experiment_name}/*.py"
            args.config = glob.glob(config_file, recursive=False)[0]
        except IndexError:
            args.config = DEFAULT_CONFIG

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.repo_id is not None:
        for hook in cfg.custom_hooks:
            if 'repo_id' in hook:
                hook['repo_id'] = args.repo_id

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # ========================================================================
    cfg.resume = args.resume
    cfg.load_from = args.load_from
    cfg.experiment_name = args.experiment_name

    # Modify config before instantiating runner
    for metric in cfg.val_evaluator.metrics:
        if args.mode == 'train':
            metric['plot_every_run'] = True
        elif args.mode == 'val':
            metric['plot_every_run'] = False

    runner = Runner.from_cfg(cfg)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'val':
        runner.val()
    elif args.mode == 'predict':
        runner.test()


if __name__ == '__main__':
    main()
