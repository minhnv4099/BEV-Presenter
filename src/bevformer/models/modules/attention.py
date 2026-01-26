#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from src.utils.logging import getLogger
from src.registry import ATTENTIONS
from src.utils.telemetry import timing
from ..utils.pytorch_utils import (
    find_prunable_heads_and_indices,
    prune_linear_layer,
)
from ..activations import ACT2FN

logger = getLogger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_mask: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
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


@ATTENTIONS.register_module()
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

    @timing
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        output_attention: bool = False
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        view_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

        queries: torch.Tensor = self.q_proj(hidden_states).view(*view_shape).transpose(1, 2)

        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states

        keys: torch.Tensor = self.k_proj(current_states).view(*view_shape).transpose(1, 2)
        values: torch.Tensor = self.v_proj(current_states).view(*view_shape).transpose(1, 2)

        context, attention_weights = eager_attention_forward(
            module=self,
            query=queries,
            key=keys,
            value=values,
            attention_mask=attention_mask,
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
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @timing
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BaseAttention(nn.Module):
    """Multihead self-attention -> Output projection"""
    def __init__(self, config):
        super().__init__()
        self.attention = BaseSelfAttention(config)
        self.output = BaseSelfOutput(config)
        self.pruned_heads = set()

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
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states)
        output = self.output(self_attn_output)
        return output


class BaseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ff_layer1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.ff_layer2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ff_layer1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.ff_layer2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states
