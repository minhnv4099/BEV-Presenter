#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import warnings
import copy
import torch
from torch import Tensor
from typing import Optional, Sequence
import torch.nn as nn

from src.typing_ import ConfigType
from src.bevformer.transformers.layers import TransformerLayerSequence
from src.bevformer.models.modules.base_transformer_layer_v2 import CustomBaseTransformerLayer
from src.registry import TRANSFORMER_BLOCKS, TRANSFORMER_LAYERS
from src.utils.logging import getLogger

logger = getLogger(__name__)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_BLOCKS.register_module()
class DetectionTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate: bool = True, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        *args,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        attn_masks: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        cls_branches: Optional[nn.ModuleList] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            key (Tensor):
                Key embeddings with shape `(bs, n_query, embed_dims)`.
            value (Tensor):
                Value embeddings with shape `(bs, n_query, embed_dims)`.
            query_pos (Tensor):
                Positional encodings for query. Default to None.
            key_pos (Tensor):
                Positional encodings for key. Default to None.
            reference_points (Tensor): The reference
                points sigmoid of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape (bs, num_query, 2).
            reg_branches (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
            cls_branches (obj:`nn.ModuleList`): List classifiers
                used by every layers. Default to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            tuple[Tensor]: Tuple of 2 tensor: final query embeddings and reference points:

                - When ``self.return_intermediate`` is False:
                    `[1, num_query, bs, embed_dims]` and `[1, bs, num_query, 2]`.
                - When ``self.return_intermediate`` is True:
                    `[n_dec_layer, num_query, bs, embed_dims]` and `[n_dec_layer, bs, num_query, 2]`.
        """
        output = query
        intermediate_hidden_states: list[Tensor] = []
        intermediate_reference_points: list[Tensor] = []
        for lid, layer in enumerate(self.layers):
            # (bs, num_query, num_level, 2)
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key,
                value,
                *args,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
            # (bs, num_query, embed_dim)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                # like offset w.r.t. reference point
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                # update reference points after a layer
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # block gradient from the last reference point
                reference_points = new_reference_points.detach()

            # (num_query, bs, embed_dim)
            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate_hidden_states.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return (torch.stack(intermediate_hidden_states),
                    torch.stack(intermediate_reference_points))

        return output.unsqueeze(0), reference_points.unsqueeze(0)


@TRANSFORMER_LAYERS.register_module()
class DetrTransformerDecoderLayer(CustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        attn_cfgs: ConfigType = dict(
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            batch_first=True),
        ffn_cfgs: ConfigType = dict(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.
        ),
        norm_cfg: ConfigType = dict(type='LN'),
        act_cfg=dict(type='ReLU', inplace=True),
        operation_order: Optional[Sequence[str]] = None,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        ffn_num_fcs: int = 2,
        **kwargs
    ):
        CustomBaseTransformerLayer.__init__(
            self,
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            norm_cfg=norm_cfg,
            operation_order=operation_order,
            act_cfg=act_cfg,
            **kwargs
        )

        self.fp16_enabled = False

        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query: Tensor,
                key: Optional[Tensor] = None,
                value: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                *,
                attn_masks: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None,
                self_attn_mask: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, \
                f'The length of attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for operation in self.operation_order:
            if operation == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index])
                attn_index += 1
                identity = query

            elif operation == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif operation == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    # level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif operation == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
