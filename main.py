#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from torch.optim import Optimizer
import torch.nn as nn
from mmengine.config import ConfigDict
from src.bevformer.builder import build_detector, build_dataset, build_optimizer
from src.bevformer.configs.bevformer_tiny_test import model as model_cfg
from src.bevformer.configs.bevformer_tiny_test import data as data_cfg
from src.bevformer.configs.bevformer_tiny_test import optimizer as opt_cfg
from src.registry import OPTIMIZERS


def main():
    from src.runner.training_args import TrainingArguments
    from src.runner.trainer import Trainer

    data_config = ConfigDict(data_cfg)
    detector_cfg = ConfigDict(model_cfg)

    dataset = build_dataset(data_config.train)
    detector: nn.Module = build_detector(detector_cfg)
    opt_cfg['params'] = detector.parameters()
    optimizer: Optimizer = build_optimizer(opt_cfg)

    args = TrainingArguments()

    trainer = Trainer(
        model=detector,
        args=args,
        train_data=dataset,
        optimizer=optimizer
    )

    trainer.run()


if __name__ == "__main__":
    main()
