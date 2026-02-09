#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    epochs: int = 10
