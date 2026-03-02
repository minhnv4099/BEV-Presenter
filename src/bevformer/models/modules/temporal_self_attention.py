#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import warnings
from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule, constant_init, xavier_init
from src.device import get_device
from src.bevformer.transformers.attentions import multi_scale_deformable_attn_pytorch
from src.utils.logging import getLogger
from src.utils.telemetry import timing
from src.registry import ATTENTIONS
from src.typing import ConfigType

logger = getLogger(__name__)


@ATTENTIONS.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature maps used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        num_bev_queue (int): In this version, we only use one history BEV and one current BEV query.
            So the length of BEV queue is 2.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout_p (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            `(bs, n, embed_dim)` or `(n, batch, embed_dim)`.
            Default to ``True``.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_bev_queue: int = 2,
        im2col_step: int = 64,
        dropout_p: float = 0.1,
        batch_first: bool = True,
        norm_cfg: Optional[ConfigType] = None,
        init_cfg: Optional[ConfigType] = None
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, '
                f'but got {embed_dims} and {num_heads}'
            )

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False
        self._is_init = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n: int):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return n != 0 and (n & (n - 1) == 0)

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set `embed_dims` in "
                '`MultiScaleDeformAttention` to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        # linear projection to compute offset
        self.sampling_offsets = nn.Linear(
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points * 2)
        # linear projection to computer attention weights instead of
        # dot product between query and key
        # because key is dynamic
        self.attention_weights = nn.Linear(
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points)
        # query is used to compute offset only
        # no for attention computing, so no need query_proj
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout_p)  # type: ignore

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            device=get_device(),
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @timing('TemporalSelfAttention')
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        atten_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor):
                The input query with shape `[num_queries, bs, embed_dims]`.
                if self.batch_first is False, else `[bs, num_queries, embed_dims]`.
            key (Tensor): The key tensor with shape
                `(bs * 2, num_value, embed_dims)`. It isn't used in this attention module.
            value (Tensor): The value tensor with shape `(num_value, bs, embed_dims)`
                if self.batch_first is False, else `[bs * 2, num_value, embed_dims]`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default None.
            reference_points (Tensor): The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                `(bs*2, num_bew_query, num_bev_level [1], 2)`
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape `[num_query, bs, embed_dims]`.
        """
        # if value is None:
        #     assert self.batch_first
        #     bs, num_query, c = query.shape
        #     value = torch.stack([query, query], 1).reshape(bs*2, num_query, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        # (bs, num_query, embed_dims*2)
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # (bs*num_bev_queue, num_value, num_head, head_dim)
        value = value.reshape(bs*self.num_bev_queue, num_value, self.num_heads, -1)

        # (bs, num_query, num_bev_queue * num_heads * num_levels * num_points * 2)
        # (bs, num_query, num_heads, num_bev_queue, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)

        # (bs, num_query, num_bev_queue * num_heads * num_levels * num_points)
        # (bs, num_query, num_heads, num_bev_queue, num_levels * num_points)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        # weights over sampling points and levels
        attention_weights = attention_weights.softmax(-1)

        # (bs, num_query, num_heads, num_bev_queue, num_levels, num_points)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)

        # (bs, n_bev_queue, num_query, num_heads, n_levels, n_points, 2)
        # (bs*n_bev_queue, num_query, num_heads, n_levels, n_points, 2)
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6) \
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # (bs, num_bev_queue, num_query, num_heads, num_levels, num_points)
        # (bs*num_bev_queue, num_query, num_heads, num_levels, num_points)
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()

        if reference_points.shape[-1] == 2:
            # size of bev plane
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # normalized sampling locations
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # two-stage
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            # if value.dtype == torch.float16:
            #     MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # else:
            #     MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            # output = MultiScaleDeformableAttnFunction.apply(
            #     value, spatial_shapes, level_start_index, sampling_locations,
            #     attention_weights, self.im2col_step)

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        # (num_query, embed_dims, bs)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            # change back (, bs, ...) as input
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
