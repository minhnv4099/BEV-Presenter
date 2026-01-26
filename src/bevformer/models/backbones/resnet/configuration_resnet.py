#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
"""ResNet model configuration"""

from typing import Optional
from dataclasses import dataclass

from src.utils.logging import getLogger
from src.registry import CONFIGS
from src.dataclass_as_dict import DataclassAsDict

logger = getLogger(__name__)


@CONFIGS.register_module()
@dataclass
class ResNetConfig(DataclassAsDict):
    r"""
    This is the configuration class to store the configuration of a [`ResNetModel`]. It is used to instantiate an
    ResNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ResNet
    [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`list[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`list[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"bottleneck"`):
            The layer to use, it can be either `"basic"` (used for smaller models, like resnet-18 or resnet-34) or
            `"bottleneck"` (used for larger models like resnet-50 and above).
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.
        downsample_in_bottleneck (`bool`, *optional*, defaults to `False`):
            If `True`, the first conv 1x1 in ResNetBottleNeckLayer will downsample the inputs using a `stride` of 2.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example::

    ```python
    >>> from transformers import ResNetConfig, ResNetModel

    >>> # Initializing a ResNet resnet-50 style configuration
    >>> configuration = ResNetConfig()

    >>> # Initializing a model (with random weights) from the resnet-50 style configuration
    >>> model = ResNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "resnet"
    layer_types = ["basic", "bottleneck"]

    def __init__(
        self,
        num_channels: Optional[int] = 3,
        embedding_size: Optional[int] = 64,
        hidden_sizes: Optional[list[int]] = [256, 512, 1024, 2048],
        depths: Optional[list[int]] = [3, 4, 6, 3],
        layer_type: Optional[str] = "bottleneck",
        hidden_act: Optional[str] = "relu",
        downsample_in_first_stage: Optional[bool] = False,
        downsample_in_bottleneck: Optional[bool] = False,
        out_features: Optional[list[int]] = None,
        out_indices: Optional[list[int]] = [2, 3, 4],
        return_dict: bool = True,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")

        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
        self.downsample_in_bottleneck = downsample_in_bottleneck
        self.stage_names = ["stem"] + [f"stage_{idx}" for idx in range(1, len(depths) + 1)]
        self.return_dict = return_dict
        self.output_hidden_states = output_hidden_states
        self.out_indices = out_indices
        # self._out_features, self._out_indices = get_aligned_output_features_output_indices(
        #     out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        # )
        super().__init__(**self.__dict__)


__all__ = ["ResNetConfig"]
