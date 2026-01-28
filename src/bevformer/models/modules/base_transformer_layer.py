#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import math
from typing import Optional, TYPE_CHECKING
import numpy as np
import copy

import torch
import torch.nn as nn

from mmengine.model import BaseModule

from src.utils.logging import getLogger
from src.registry import TRANSFORMER_LAYERS, MODELS, TRANSFORMER_BLOCKS
from src.modeling_output import BaseModelOutput
from src.utils.telemetry import timing
from src.bevformer.models.utils.bricks import build_transformer_layer
from src.typing import ConfigType
from .base_attention import BaseAttention, BaseFeedForward
from .interpolation import InterpolateMidPositionEmbeddings

if TYPE_CHECKING:
    ...

logger = getLogger(__name__)


@MODELS.register_module()
class PatchEmbeddings(nn.Module):
    """ Construct the position embedding into feature maps.
    Embedding layer receiving feature maps and make it ready to feed to Transformers.
    """


@TRANSFORMER_LAYERS.register_module()
class BaseOriginalTransformerLayer(BaseModule):
    """This corresponds to the Block class in the timm implementation.

    Consisting of `Attention` -> `FeedForward`.
    """

    def __init__(self, config):
        super().__init__()

        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BaseAttention(config)
        self.ffn = BaseFeedForward(config)

        self.layernorm_before = nn.LayerNorm(config['hidden_size'], eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.embed_dims = config.hidden_size
        self.pre_norm = True

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attention: bool = False,
        **kwargs
    ) -> torch.Tensor:
        identity = query

        # layernorm unnormalized input
        query_norm = self.layernorm_before(query)
        # first residual connection occurs in attention layer
        attention_output = self.attention(
            query_norm,
            key,
            value,
            identity if self.pre_norm else None,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            head_mask=head_mask,
            **kwargs
        )

        # layernorm is also applied after self-attention
        hidden_states = self.layernorm_after(attention_output)
        # second residual connection is done ffn layer
        layer_output = self.ffn(hidden_states, hidden_states)

        return layer_output


@TRANSFORMER_BLOCKS.register_module()
class BaseTransformerEncoder(nn.Module):
    """The consecutive Encoder layers."""

    def __init__(self, config, layer_configs: Optional[ConfigType] = None) -> None:
        super().__init__()

        self.config = config
        self.gradient_checkpointing = False

        if layer_configs is not None:
            if isinstance(layer_configs, dict):
                layer_configs['config'] = layer_configs.get('config', config)
                layer_configs = [layer_configs for _ in range(config.num_hidden_layers)]
            assert len(layer_configs) == config.num_hidden_layers

            self.layers = nn.ModuleList()
            for layer_config in layer_configs:
                self.layers.append(build_transformer_layer(layer_config))
        else:
            self.layers = nn.ModuleList([BaseOriginalTransformerLayer(config) for _ in range(config.num_hidden_layers)])

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

    @timing(scope='BaseTransformerEncoder')
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> BaseModelOutput:
        hidden_states, h, w = self.flatten_patches_to_sequence(query)
        height = h * self.config.patch_size
        width = w * self.config.patch_size

        interpolated_mid_position_embeddings = None
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

        for i, layer_module in enumerate(self.layers):
            # Use mid position embeddings to 0 -> 10, except for the last layer
            if interpolated_mid_position_embeddings is not None:
                if i < (self.config.num_hidden_layers - 1):
                    query_pos = interpolated_mid_position_embeddings[i]

            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_mask=attn_mask,
                head_mask=layer_head_mask,
                **kwargs
            )

        return BaseModelOutput(last_hidden_state=hidden_states)
