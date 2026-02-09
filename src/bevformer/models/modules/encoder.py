#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import copy
import warnings
from typing import Optional, Literal, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.version import __version__ as TORCH_VERSION

from src.registry import (
    TRANSFORMER_LAYERS,
    TRANSFORMER_BLOCKS
)
from mmengine.utils.version_utils import digit_version

from src.utils.logging import getLogger
from src.utils.fp16_utils import force_fp32, auto_fp16
from src.typing import ConfigType
from src.bevformer.transformers.layers import TransformerLayerSequence
from .base_transformer_layer_v2 import CustomBaseTransformerLayer

logger = getLogger(__name__)


@TRANSFORMER_BLOCKS.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    """
    Attention with both self and cross. Implements the decoder in DETR transformer.
    Args:
        pc_range (Sequence[int | float]): Point clouds range.
        return_intermediate (bool): Whether to return intermediate outputs. Used for refinement.
        num_points_in_pillar (int): Number of sampling points in pillar at a location in bev plane.
        coder_norm_cfg (dict): Config of last normalization layer. Default to`LN`.
        transformerlayers (ConfigType): Config to construct transformer layers.
        num_layers (int): Number of transformer layers in that block (sequence).
    """

    def __init__(
        self,
        pc_range: Optional[Sequence[int | float]] = None,
        num_points_in_pillar: int = 4,
        return_intermediate: bool = False,
        transformerlayers: Optional[ConfigType] = None,
        num_layers: int = 6,
        dataset_type: str = 'nuscenes',
        **kwargs
    ):
        super(BEVFormerEncoder, self).__init__(
            transformerlayers=transformerlayers,
            num_layers=num_layers,
            **kwargs
        )

        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(
        H: int, W: int,
        Z=8, bs=1,
        num_points_in_pillar=4,
        dim: Literal['3d', '2d'] = '3d',
        device='cpu',
        dtype=torch.float
    ) -> Tensor:
        """Get the reference points used in SCA and TSA.
        The points are range 0 -> 1 relative and uniform to bev plane size.
        Args:
            H: Height of bev plane.
            W: Width of bev plane.
            Z: Height of a pillar.
            bs: Batch size.
            num_points_in_pillar: Sample D points uniformly in each pillar.
            device (obj:`device`): The device where ``reference_points`` should be.
            dtype (:obj:`dtype`): The type ``reference_points`` should be.
            dim: Type of reference point: 2D or 3D.
        Returns:
            Tensor of uniform reference points w.r.t. bev plane (normalized by `H`, `W`, `Z`).

                - Has shape `(bs, num_points_in_pillar, HxW, 3)` if ``3d``.
                - Has shape `(bs, HxW, 1, 2)` if ``2d``
        """
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            # shape `(num_points_in_pillar, H, W)`
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            # shape `(num_points_in_pillar, H, W, 3)`
            ref_3d = torch.stack((xs, ys, zs), -1)
            # `(num_points_in_pillar, 3, H, W)`
            # -> `(num_points_in_pillar, 3, H*W)`
            # -> `(num_points_in_pillar, H*W, 3)`
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # -> `(bs, num_points_in_pillar, H*W, 3)`
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            # shape `(H, W)` both
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            # shape `(1, HxW)`
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            # shape `(1, HxW, 2)`
            ref_2d = torch.stack((ref_x, ref_y), -1)
            # shape `(bs, HxW, 2)`
            # -> `(bs, HxW, 1, 2)`
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

        return ...

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(
        self,
        reference_points: Tensor,
        pc_range: Sequence[float],
        img_metas: list[dict]
    ) -> tuple[Tensor, Tensor]:
        """Get the normalized reference points on the camera views.
        Args:
            reference_points (`Tensor`):
                The 3D relative bev reference points (0 -> 1) uniformly to bev_h, bev_w, Z.
                Results of ``self.get_reference_points``.
                Shape of `(bs, num_points_in_pillar, H*W [num_query], 3)`
            pc_range (`Sequence[float]`):
                Point cloud range.
            img_metas (`list[dict]`):
                List of `bs` of `n_cam` (metadata).
        Returns:
            Tuple of 2 tensors

                **Coordinates of projected references points on views**:
                    Projected reference points from 3D -> 2D.
                    Relative and normalized w.r.t. image shape. Used by re-scale to get coordinates w.r.t each feature level.
                    Shape `(num_cam, bs, H*W, num_points_in_pillar, 2)`.
                **BEV mask**: Contains information whether the ``point`` hit that ``cam`` in that ``sample`` at that ``level``.
                    Shape of `(num_cam, bs, H*W, num_points_in_pillar)`.
        """
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # get projection matrix from cameras
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        # shape of (bs, num_cam, 4, 4)
        lidar2img = reference_points.new_tensor(lidar2img)
        num_cam = lidar2img.size(1)

        reference_points = reference_points.clone()
        # compute real world location of reference points in bev plane, origin is center
        # each cell in bev plane cover an area in real world.
        # bev-relative * actual axis length -> real points in meter from left to right, ...
        # + negative left -> actual coordinate w.r.t. ego car whose location is the origin (0,0).
        # Example:
        # (0.1, 0.2, ... 1) * 10 meters -> (0, 1, 2, ... 10 meters)
        # (0, 1, 2, ... 10 meters) + -5 (pc_range[0]) -> (-5,-4,-3,-2,-1,0,1,2,3,4,5), `0` is location of ego car.
        # shape keep the same as input: `(bs, num_points_in_pillar, H*W [num_query], 3)`
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # stacking 1s in last dim (3 -> 4) to multiply matrix
        # shape `(bs, num_pillar_points, H*W, 4)`
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        # D = num_pillar_points
        # shape `(D, bs, H*W, 4)`
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]

        # shape `(D, bs, 1, H*W, 4)`
        # -> `(D, bs, num_cam, H*W, 4)`
        # -> `(D, bs, num_cam, H*W, 4, 1)`
        reference_points = reference_points.view(D, B, 1, num_query, 4)
        reference_points = reference_points.repeat(1, 1, num_cam, 1, 1)
        reference_points = reference_points.unsqueeze(-1)

        # shape `(1, bs, num_cam, 1, 4, 4)`
        # -> `(D, bs, num_cam, H*W, 4, 4)`
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4)
        lidar2img = lidar2img.repeat(D, 1, 1, num_query, 1, 1)

        # project real world reference points (x′ y′ z′j 1) onto the cameras
        # compute Ti · [x′ y′ z′j 1]T in formula zij · [xij yij 1]T = Ti·[x′ y′ z′j 1]T # NOTE: 1
        # (D, bs, num_cam, H*W, 4, 4) x (D, bs, num_cam, H*W, 4, 1)
        # -> `(D, bs, num_cam, H*W, 4, 1)`
        # -> `(D, bs, num_cam, H*W, 4)`
        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32),
            reference_points.to(torch.float32)
        )
        reference_points_cam = reference_points_cam.squeeze(-1)

        eps = 1e-5
        # compare on z-axis
        # z > eps, meaning the point with that height is hit view  # NOTE: need justify
        # `(D, bs, num_cam, H*W, 1)`
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        # from NOTE 1
        # divide [x y] by zij to get [xij yij] pixel coordinates in image
        # divide eps to avoid divide 0
        # -> `(D, bs, num_cam, H*W, 2)`
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # normalized projected x,y with image size
        # used to get the corresponding reference points on each feature level
        # by ratios between image and feature maps (H/12, H/24 ...)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        # attention points in view of cameras
        bev_mask = (bev_mask
                    & (reference_points_cam[..., 0:1] > 0.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    )
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # shape `(num_cam, bs, H*W, D, 2)`
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # `(num_cam, bs, H*W, D, 1)`
        # -> `(num_cam, bs, H*W, D)`
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    # TODO: require key and value when actual using
    @auto_fp16()
    def forward(
        self,
        bev_query: Tensor,
        key: Tensor,
        value: Tensor,
        *args,
        bev_h: int,
        bev_w: int,
        spatial_shapes: Tensor,
        bev_pos: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        valid_ratios: Optional[Sequence[float]] = None,
        prev_bev: Optional[Tensor] = None,
        shift: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor):
                Input BEV query with shape `(num_query, bs, embed_dims)`.
            key (Tensor):
                Input multi-camera features with shape `(num_cam, num_value, bs, embed_dims)`.
                Pass to SCA.
            value (Tensor):
                Input multi-camera features with shape typically same as ``key`` `(num_cam, num_value, bs, embed_dims)`.
                Pass to SCA.
            bev_h (int): Height of bev plane.
            bev_w (int): Width of bev plane.
            bev_pos (`Tensor`): Position embeddings of bev.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index:
            valid_ratios (Tensor):
                The radios of valid points on the feature map, has shape `(bs, num_levels, 2)`
            prev_bev (Tensor):
                The previous BEV feature with shape `(num_query, bs, embed_dims)`.
            shift (Tensor): Shape of [1, 2]
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        logger.info("Get 3D reference points in space used for spatial-cross attention (SCA)")
        # 3d reference point in space used for spatial-cross attention (SCA)
        ref_3d = self.get_reference_points(
            H=bev_h, W=bev_w, Z=self.pc_range[5] - self.pc_range[2],
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=bev_query.size(1),  
            device=bev_query.device, dtype=bev_query.dtype)
        
        logger.info("Get 2D reference points in bev plane used for temporal-self attention (TSA)")
        # 2d reference points in bev plane used for temporal-self attention (TSA)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), 
            device=bev_query.device, dtype=bev_query.dtype)

        logger.info("Get real location of reference points on camera views.")
        reference_points_cam, bev_mask = self.point_sampling(
            reference_points=ref_3d,
            pc_range=self.pc_range,
            img_metas=kwargs['img_metas']
        )

        if shift is None:
            shift = torch.scalar_tensor(0., requires_grad=False)
        else:
            if isinstance(shift, float):
                shift = torch.scalar_tensor(shift, requires_grad=False),
                
        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # align ref point 2d to match with previous bev instead of aligning previous bev. NOTE need to justify.
        logger.info("Align BEV plane by 2D reference points.")
        shift_ref_2d = ref_2d.clone() + shift[:, None, None, :]
    
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2) if bev_pos is not None else None

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for layer in self.layers:
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                reference_points_cam=reference_points_cam,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYERS.register_module()
class BEVFormerLayer(CustomBaseTransformerLayer):
    """Implements a [decoder] layer in blocks.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
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
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """
    default_operation_order = {'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'}

    def __init__(
        self,
        attn_cfgs: Optional[ConfigType] = None,
        ffn_cfgs: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='LN'),
        operation_order: Optional[Sequence[str]] = None,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        ffn_num_fcs: int = 2,
        act_cfg=dict(type='ReLU', inplace=True),
        **kwargs
    ):
        if operation_order is None:
            operation_order = operation_order or self.default_operation_order
            warnings.warn(f"Used default operation order: {operation_order}.")

        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            norm_cfg=norm_cfg,
            operation_order=operation_order,
            act_cfg=act_cfg,
            # feedforward_channels=feedforward_channels,
            # ffn_dropout=ffn_dropout,
            # ffn_num_fcs=ffn_num_fcs,
            **kwargs
        )
        self.fp16_enabled = False

        # based on the original BEVFormer paper
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        *args,
        bev_pos: Optional[Tensor] = None,
        # query_pos: Optional[Tensor] = None,
        # key_pos: Optional[Tensor] = None,
        attn_masks: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        ref_2d: Optional[Tensor] = None,
        ref_3d: Optional[Tensor] = None,
        bev_h: Optional[int] = None,
        bev_w: Optional[int] = None,
        reference_points_cam: Optional[Tensor] = None,
        spatial_shapes=None,
        level_start_index=None,
        bev_mask: Optional[Tensor] = None,
        prev_bev: Optional[Tensor] = None,
        **kwargs
    ):
        """Forward function for `TransformerDecoderLayer`.

        Args:
            query (Tensor):
                The input query with shape [num_queries, bs, embed_dims]
                if self.batch_first is False, else `[bs, num_queries embed_dims]`.
            key (Tensor):
                Input multi-camera features with shape `(num_cam, num_value, bs, embed_dims)`.
            value (Tensor):
                Input multi-camera features with shape typically same as ``key``.
            bev_pos (Tensor): The positional encoding for `bev`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            ref_2d (Tensor):
                2D reference points in bev plane used for temporal-self attention (TSA).
                Shape `(bs*2, num_bew_query, num_bev_level [1], 2)`
            ref_3d (Tensor):
                3D reference point in space used for spatial-cross attention (SCA).
                Shape `(bs, D, num_bew_query, 3)`, `D = num_points_in_pillar`.
            bev_h (int): Height of bev plane.
            bev_w (int): Width of bev plane.
            reference_points_cam (Tensor):
                Normalized reference points on camera view.
                Shape `(num_cam, bs, H*W, num_points_in_pillar, 2)`.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
            spatial_shapes (Tensor): Spatial shape of feature maps in
                different levels. With shape `(num_levels, 2)`,
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape `(num_levels, )` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            bev_mask (Tensor): Contains information that whether the point hit that cam in that sample at that level`
                    Shape of `(num_cam, bs, H*W, D)`.
            prev_bev (Tensor):
                Tensor of previous bev and current bev query. Or ``None``.
                The bev feature of previous timestep. Shape `(bs*2, num_query, embed_dims)`.
            **kwargs: Some specific arguments of attentions.

        Returns:
            Tensor: forwarded results with shape `[num_queries, bs, embed_dims].`
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, \
                f'The length of attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    reference_points=ref_2d,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=Tensor([0], device=query.device),
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spatial cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    # query_pos=query_pos,
                    # key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYERS.register_module()
class MMBEVFormerLayer(CustomBaseTransformerLayer):
    """Multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        operation_order = operation_order or BEVFormerLayer.default_operation_order

        super(MMBEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        self.fp16_enabled = False

        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(Tensor(0.5), requires_grad=True)

        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                feature_spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    feature_spatial_shapes=Tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=Tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    feature_spatial_shapes=feature_spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        feature_spatial_shapes=Tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=Tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
