#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch
import numpy as np


def to_tensor(data: list):
    return torch.as_tensor(np.array(data))
