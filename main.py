#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch
from torch.optim import Optimizer
import torch.nn as nn
from mmengine.config import ConfigDict, Config
from src.bevformer.builder import build_detector, build_dataset, build_optimizer
from src.bevformer.configs.bevformer_tiny_test import model as model_cfg
from src.bevformer.configs.bevformer_tiny_test import data as data_cfg
from src.bevformer.configs.bevformer_tiny_test import optimizer as opt_cfg
from src.datasets.builder import build_dataloader


def main():
    from src.runner.runner import Runner
    cfg_file = "src/bevformer/configs/bevformer_tiny_test.py"
    cfg = Config.fromfile(cfg_file)
    data_config = ConfigDict(data_cfg)
    detector_cfg = ConfigDict(model_cfg)

    train_dataset = build_dataset(data_config.train)
    val_dataset = build_dataset(data_config.train)
    detector: nn.Module = build_detector(detector_cfg)

    opt_cfg['params'] = detector.parameters()
    optimizer: Optimizer = build_optimizer(opt_cfg)
    train_dataloader = build_dataloader(train_dataset, batch_size=1)
    val_dataloader = build_dataloader(val_dataset, batch_size=1)

    val_kwargs = dict(
        val_dataloader=val_dataloader,
        val_cfg=dict(by_epoch=True, max_epochs=10),
    )
    runner = Runner(
        model=detector,
        work_dir='experiment',
        train_dataloader=train_dataloader,
        optim_wrapper=dict(optimizer=optimizer),
        train_cfg=dict(by_epoch=True, max_epochs=10),
        resume=True,
        cfg=cfg,
    )
    runner.register_default_hooks()
    runner.load_or_resume()


if __name__ == "__main__":
    main()
