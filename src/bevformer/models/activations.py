#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch.nn.functional as F
import torch.nn as nn


ACT2FN = {
    "relu": nn.ReLU(),
    "softmax": nn.Softmax(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    None: nn.Identity()
}
