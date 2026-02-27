#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            `(bs, num_values, num_heads, embed_dims//num_heads)`
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape `(num_levels, 2)`,
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            `(bs ,num_queries, num_heads, num_levels, num_points, 2)`,
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            `(bs, num_queries, num_heads, num_levels, num_points)`,

    Returns:
        `torch.Tensor` has shape `(bs, num_queries, embed_dims)`
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # split value to list of value each value corresponds to each level feature
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # (bs, H_*W_, num_heads, embed_dims) ->
        # (bs, H_*W_, num_heads*embed_dims) ->
        # (bs, num_heads*embed_dims, H_*W_) ->
        # (bs*num_heads, embed_dims, H_, W_)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # (bs, num_queries, num_heads, num_points, 2) ->
        # (bs, num_heads, num_queries, num_points, 2) ->
        # (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # (bs*num_heads, embed_dims, num_queries, num_points)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention scores.

    Args:
        module:
            Module, used to check whether the module is training to apply dropout properly.
        query:
            Query shape `(batch_size, num_heads, n_query, dim)`
        key:
            Key shape `(batch_size, num_heads, n_key, dim)`
        value:
            Value shape `(batch_size, num_heads, n_value, dimV)`. Most cases `dimV` = `dim`.
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
        tuple[Tensor]: Tuple of:

            - attention hidden states `(batch_size, q_length, num_heads, attention_head_size)`
            - attention weights `(batch_size, num_heads, q_length, k_length)`.
    """
    # (batch_size, num_attention_heads, q_length, k_length)
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=query.dtype).to(query.dtype)

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
