# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import copy
import warnings
from typing import Optional, Union

import torch
from torch import Tensor
import torch.nn as nn

from mmengine.model import BaseModule, ModuleList

from src.registry import TRANSFORMER_LAYERS
from src.bevformer.models.utils.bricks import build_feedforward_network, build_attention, build_norm_layer
from src.typing import ConfigType

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))
except ImportError:
    # warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
    #               '``mmcv.ops.multi_scale_deform_attn``, '
    #               'You should install ``mmcv-full`` if you need this module. ')
    pass


@TRANSFORMER_LAYERS.register_module()
class CustomBaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.
    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None ):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in `operation_order`.
            If it is a dict, all of the attention modules in `operation_order`
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in `operation_order`.
            If it is a dict, all of the attention modules in `operation_order`
            will be built with this config.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        act_cfg:
        operation_order (tuple[str]): The execution order of operation
            in the layer. Such as (`self_attn`, `norm`, `ffn`, `cross_attn`, `norm`, `ffn).
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """
    available_operations: set[str] = {'self_attn', 'norm', 'ffn', 'cross_attn'}

    def __init__(
        self,
        attn_cfgs: Optional[ConfigType] = None,
        ffn_cfgs: ConfigType = dict(
            type='FFN',
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.,
        ),
        norm_cfg: ConfigType = dict(type='LN'),
        act_cfg=dict(type='ReLU', inplace=True),
        operation_order: Optional[tuple[str]] = None,
        init_cfg: ConfigType = None,
        batch_first: bool = True,
        **kwargs
    ):
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs'
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(CustomBaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        if operation_order is None:
            operation_order = self.operation_order
        assert set(operation_order) and set(operation_order) == self.available_operations,  \
            f'The operation_order of {self.__class__.__name__} should ' \
            f'contains all four operation types: "{self.available_operations}"'

        # duplicate attention configs based on 'self_attn' and 'cross_attn' in the operation order set
        self.num_attn = num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        assert attn_cfgs
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        assert num_attn == len(attn_cfgs), \
            f'The length of attn_cfg {num_attn} is ' \
            f'not consistent with the number of attention' \
            f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first

                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims: int = self.attentions[0].embed_dims

        num_ffns = operation_order.count('ffn')
        self.ffns = ModuleList()
        assert ffn_cfgs
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns

        for ffn_cfg in ffn_cfgs:
            if 'embed_dims' not in ffn_cfg:
                ffn_cfg['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfg['embed_dims'] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfg))

        num_norms = operation_order.count('norm')
        self.norms = ModuleList()
        assert norm_cfg
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, num_features=self.embed_dims)[1])

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        attn_masks: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward function for `TransformerDecoderLayer`.
        Args:
            query (Tensor): The input query with shape `[num_queries, bs, embed_dims]`
                if self.batch_first is False, else `[bs, num_queries embed_dims].`
            key (Tensor): The key tensor with shape `[num_keys, bs, embed_dims]`
                if self.batch_first is False, else `[bs, num_keys, embed_dims] .`
            value (Tensor): The value tensor with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
            **kwargs: Some specific arguments of attentions.

        Returns:
            Tensor: forwarded results with shape `[num_queries, bs, embed_dims]`.
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
                          f'{self.__class__.__name__!r} ')
        else:
            assert len(attn_masks) == self.num_attn, \
                f'The length of attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
