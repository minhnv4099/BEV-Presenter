#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import math
from typing import Optional
import numpy as np
import copy

import torch
import torch.nn as nn

from mmengine.model import BaseModule

from src.utils.logging import getLogger
from src.registry import TRANSFORMER_LAYERS, MODELS
from src.modeling_output import BaseModelOutput
from src.utils.telemetry import timing
from .attention import BaseAttention, BaseFeedForward
from .interpolation import InterpolateMidPositionEmbeddings

logger = getLogger(__name__)


@MODELS.register_module()
class PatchEmbeddings(nn.Module):
    """ Construct the position embedding into feature maps.
    Embedding layer receiving feature maps and make it ready to feed to Transformers.
    """


@TRANSFORMER_LAYERS.register_module()
class BaseTransformerLayer(BaseModule):
    """This corresponds to the Block class in the timm implementation.

    Consisting of `Attention` -> `FeedForward`.
    """

    def __init__(self, config):
        super().__init__()

        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BaseAttention(config)
        self.output = BaseFeedForward(config)

        self.layernorm_before = nn.LayerNorm(config['hidden_size'], eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, attention_mask, head_mask, encoder_hidden_states)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Yolos, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


@MODELS.register_module()
class BaseTransformerEncoder(nn.Module):
    """The consecutive Encoder layers."""

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.layer = nn.ModuleList([BaseTransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        seq_length = (config.image_size[0] * config.image_size[1]) // config.patch_size**2

        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )

        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

    def flatten_patches_to_sequence(self, patches: torch.Tensor):
        if len(patches.shape) == 3:
            seq_len = patches.size(1)
            return patches, int(math.sqrt(seq_len)), int(math.sqrt(seq_len))

        # len(patches.shape) == 4:
        batch_size, hidden_size, h, w = patches.size()
        patches = patches.permute(0, 2, 3, 1).view(batch_size, h * w, hidden_size)
        return patches, h, w

    @timing
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutput:
        hidden_states, h, w = self.flatten_patches_to_sequence(hidden_states)
        height = h * self.config.patch_size
        width = w * self.config.patch_size

        interpolated_mid_position_embeddings = None
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

        for i, layer_module in enumerate(self.layer):
            if self.config.use_mid_position_embeddings:
                # Use mid position embeddings to 0 -> 10, except for the last layer
                if interpolated_mid_position_embeddings is not None:
                    if i < (self.config.num_hidden_layers - 1):
                        hidden_states = hidden_states + interpolated_mid_position_embeddings[i]

            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


