#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import warnings
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from src.utils.logging import getLogger
from src.registry import ATTENTIONS, MODELS
from src.utils.telemetry import timing
from ..utils.pytorch_utils import (
    find_prunable_heads_and_indices,
    prune_linear_layer,
)
from ..activations import ACT2FN
from .configs import BaseTransformerConfig

logger = getLogger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention scores.

    Args:
        module:
            Module, used to check whether the module is training to apply dropout properly.
        query:
            Query shape `(batch_size, num_heads, q_length, dim)`
        key:
            Key shape `(batch_size, num_heads, k_length, dim)`
        value:
            Value shape `(batch_size, num_heads, k_length, dimV)`. Most cases `dimV` = `dim`.
        head_mask:
            Mask of `1s` and `0s` to ignore some heads after computing attention weights.
            Shape `(num_layers, num_heads)` or `(num_heads, )`
        attention_mask:
            Mask to ignore some token before computing attention weights.
            Shape `(batch_size, 1, seq_length)` or `(batch_size, 1, q_length, k_length)`
        scaling:
            Scaling factor before computing softmax.
        dropout:
            Dropout probability after computing softmax.

    Returns:
        Tuple of attention hidden states `(batch_size, q_length, num_attention_heads, attention_head_size)`
        and attention weights `(batch_size, num_attention_heads, q_length, k_length)`.
    """
    # Take the dot product between "query" and "key" to get the raw attention scores.
    # (batch_size, num_attention_heads, q_length, k_length)
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Mask token if we want to
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # (batch_size, num_attention_heads, q_length, attention_head_size)
    attn_output = torch.matmul(attn_weights, value)
    # (batch_size, q_length, num_attention_heads, attention_head_size)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BaseSelfAttention(BaseModule):
    relative_position_embedding_types = ("relative_key", "relative_key_query")

    def __init__(
        self,
        config,
        position_embedding_type: Optional[str] = None
    ):
        BaseModule.__init__(self)
        self.config = config

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.q_proj = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.k_proj = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.v_proj = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)

        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type or getattr(
            config, 'position_embedding_type', "absolute"
        )

        if self.position_embedding_type in self.relative_position_embedding_types:
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attention: bool = False,
        **kwargs
    ):
        batch_size, seq_len, hidden_size = query.shape
        view_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

        query: torch.Tensor = self.q_proj(query).view(*view_shape).transpose(1, 2)
        key: torch.Tensor = self.k_proj(key).view(*view_shape).transpose(1, 2)
        value: torch.Tensor = self.v_proj(value).view(*view_shape).transpose(1, 2)

        context, attention_weights = eager_attention_forward(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attn_mask or attention_mask,
            head_mask=head_mask,
            scaling=1,
            dropout=self.attention_probs_dropout_prob
        )
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(context_shape)

        return context, attention_weights


class BaseSelfOutput(nn.Module):
    """
    The output project to transform concatenated head attentions for multihead attention layer.

    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


@ATTENTIONS.register_module()
class BaseAttention(nn.Module):
    """Multihead self-attention -> Output projection"""

    def __init__(self, config: Optional[BaseTransformerConfig] = None, **kwargs):
        super().__init__()

        config = config or BaseTransformerConfig()
        self.attention = BaseSelfAttention(config)
        self.output_proj = BaseSelfOutput(config)
        self.pruned_heads = set()

        self.embed_dims = config.hidden_size

    def prune_heads(self, heads: set[int]):
        if len(heads) == 0:
            return
        heads, index = find_prunable_heads_and_indices(
            heads=heads,
            n_heads=self.attention.num_attention_heads,
            head_size=self.attention.attention_head_size,
            already_pruned_heads=self.pruned_heads
        )

        # Prune linear layers
        self.attention.q_proj = prune_linear_layer(self.attention.q_proj, index)
        self.attention.k_proj = prune_linear_layer(self.attention.k_proj, index)
        self.attention.v_proj = prune_linear_layer(self.attention.v_proj, index)
        self.output_proj.output_proj = prune_linear_layer(self.output_proj.output_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @timing(scope="BaseAttention")
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attention: bool = False,
        **kwargs
    ) -> torch.Tensor:
        key = key if key is not None else query
        value = value if value is not None else key
        key_pos = key_pos if key_pos is not None else query_pos

        if query_pos is not None:
            if query_pos.dim() == query.dim() - 1:
                query += query_pos[None]
            else:
                query += query_pos
        else:
            warnings.warn("`query_pos` is missing.")

        if key_pos is not None:
            if key_pos.dim() == key.dim() - 1:
                key += key_pos[None]
            else:
                key += key_pos
        else:
            warnings.warn("`key_pos` is missing.")

        attn_output, _ = self.attention(
            query,
            key,
            value,
            # query_pos,
            # key_pos,
            attn_mask=attn_mask,
            head_mask=head_mask,
            **kwargs
        )
        output = self.output_proj(attn_output)

        if identity is not None:
            output = output + identity

        return output


@MODELS.register_module()
class BaseFeedForward(nn.Module):
    def __init__(self, config: Optional[BaseTransformerConfig] = None, **kwargs):
        super().__init__()
        config = config or BaseTransformerConfig()

        self.ff_layer1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.ff_layer2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob, inplace=True)

    def forward(self, hidden_states: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.ff_layer1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.ff_layer2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if identity is not None:
            hidden_states = hidden_states + identity

        return hidden_states
