#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch
import torch.nn as nn

from src.utils.fp16_utils import auto_fp16
from src.registry import MODELS


@MODELS.register_module()
class BaseDownChannel(nn.Module):
    def __init__(
        self,
        in_channels: list[int] = (512, 1024, 2048),
        out_channel: int = 256,
        conv_bias: bool = False
    ):
        super().__init__()

        for in_channel in in_channels:
            conv = nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel,
                kernel_size=1, stride=1,
                bias=conv_bias
            )
            setattr(self, f'conv1x1_{in_channel}', conv)

        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels[-1],
            out_channels=out_channel,
            kernel_size=3,
            stride=2,
            bias=conv_bias
        )

    @auto_fp16(apply_to=())
    def forward(self, feature_maps: list[torch.FloatTensor]):
        outputs = []
        for feat in feature_maps:
            channel = feat.shape[-3]
            conv = eval(f"self.conv1x1_{channel}")
            outputs.append(conv(feat))

        outputs.append(self.conv3x3(feature_maps[-1]))

        return outputs
