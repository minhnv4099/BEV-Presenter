#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Union
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from .training_args import TrainingArguments
from src.utils.logging import getLogger

logger = getLogger(__name__)


class Trainer:

    def __init__(self,
                model: nn.Module,
                args: TrainingArguments,
                train_data: Union[Dataset, DataLoader],
                optimizer: Optimizer,
                val_data: Union[DataLoader, Dataset] = None,
    ):

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.args = args
        self.optimizer = optimizer

    def training_step(self):
        pass

    def run(self):
        for i, batch in enumerate(self.train_data):
            logger.info(f"Step: {i}")
            loss = self.model(**batch)
            logger.info(loss)
            cls_loss: torch.Tensor = loss['loss_cls']
            box_loss: torch.Tensor = loss['loss_bbox']

            self.optimizer.zero_grad()
            cls_loss.backward(retain_graph=True)
            box_loss.backward(retain_graph=True)
            self.optimizer.step()
