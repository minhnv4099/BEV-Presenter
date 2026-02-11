
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from typing import Sequence
from functools import partial

import numpy as np
import torch
from mmengine.dataset.utils import default_collate
from mmengine.dist import get_dist_info
from mmengine.registry.build_functions import build_from_cfg
from torch.utils.data import DataLoader

from .samplers.group_sampler import DistributedGroupSampler, GroupSampler
from .samplers.distributed_sampler import DistributedSampler

from src.registry import DATA_SAMPLERS
from src.device import get_device


def build_dataloader(
    dataset,
    batch_size=1,
    samples_per_gpu=3,
    workers_per_gpu=0,
    num_gpus=1,
    dist=False,
    shuffle=True,
    seed=None,
    shuffler_sampler=None,
    nonshuffler_sampler=None,
    **kwargs
):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DATA_SAMPLERS.build(
                shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                default_args=dict(
                    dataset=dataset,
                    samples_per_gpu=samples_per_gpu,
                    num_replicas=world_size,
                    rank=rank,
                    seed=seed)
                )
        else:
            sampler = DATA_SAMPLERS.build(
                nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                default_args=dict(
                    dataset=dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed)
                )

        # batch_size = samples_per_gpu
        batch_size = batch_size
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        # sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        # batch_size = num_gpus * samples_per_gpu
        # num_workers = num_gpus * workers_per_gpu
        batch_size = batch_size
        num_workers = workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_func,
        pin_memory=False,
        worker_init_fn=init_fn,
        persistent_workers=(num_workers > 0),
        **kwargs)

    return data_loader


def collate_func(data_batch: Sequence[dict]):
    img = []
    img_metas = []
    gt_bboxes_3d = []
    gt_labels_3d = []

    for sample in data_batch:
        img.append(sample['img'])
        img_metas.append(sample['img_metas'])
        gt_bboxes_3d.append(sample['gt_bboxes_3d'])
        gt_labels_3d.append(sample['gt_labels_3d'])

    return {
        'img': torch.stack(img, dim=0).to(get_device()),
        'img_metas': img_metas,
        'gt_bboxes_3d': gt_bboxes_3d,
        'gt_labels_3d': gt_labels_3d
    }


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
