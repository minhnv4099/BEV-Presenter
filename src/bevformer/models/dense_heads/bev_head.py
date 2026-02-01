from __future__ import annotations

from typing import Sequence, Optional, TYPE_CHECKING
import torch
import torch.nn as nn

from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead
from mmengine.model import BaseModule

from src.typing import ConfigType
from src.registry import HEADS
from src.utils.logging import getLogger
from src.utils.fp16_utils import force_fp32, auto_fp16
from src.bevformer.builder import build_head, build_positional_encoding
from src.bevformer.models.utils.bricks import build_transformer

if TYPE_CHECKING:
    from ..modules.transformer import PerceptionTransformer

logger = getLogger(__name__)


@HEADS.register_module()
class BEVHead(BaseModule):
    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        pc_range: Sequence[float],
        embed_dims: int,
        transformer: ConfigType,
        positional_encoding: ConfigType,
        pts_bbox_head_3d: ConfigType,
        init_cfg: Optional[ConfigType] = None,
        **kwargs
    ):
        super(BEVHead, self).__init__(init_cfg=init_cfg)

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.fp16_enabled = False

        self.transformer: "PerceptionTransformer" = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        pts_bbox_head_3d.update(kwargs)
        self.pts_bbox_head_3d = build_head(pts_bbox_head_3d)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the Multi View BEV Encoder"""
        self.transformer.init_weights()

    @force_fp32(apply_to=('mlvl_feats', 'pred_bev'))
    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        img_metas: dict[str, ...],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        if only_bev:
            return bev_embed

        bev_feature = bev_embed.permute(0, 2, 1).reshape(bs, self.embed_dims, self.bev_h, self.bev_w)
        ret = dict()
        ret['pred'] = self.pts_bbox_head_3d([bev_feature, ])
        if not self.training:
            ret['bev_embed'] = bev_embed
        return ret

    @force_fp32(apply_to=('ret', ))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             ret,
             gt_bboxes_ignore=None,
             img_metas=None):
        assert gt_bboxes_ignore is None
        return self.pts_bbox_head_3d.loss(gt_bboxes_list, gt_labels_list, ret['pred'],
                                          gt_bboxes_ignore=gt_bboxes_ignore, img_metas=img_metas)

    @force_fp32(apply_to=('ret'))
    def get_bboxes(self, ret, img_metas, rescale=False):
        return self.pts_bbox_head_3d.get_bboxes(ret['pred'], img_metas)


@HEADS.register_module()
class FreeAnchor3DHeadV2(FreeAnchor3DHead):
    @force_fp32(apply_to=('pred'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             pred,
             gt_bboxes_ignore=None,
             img_metas=None):
        cls_scores, bbox_preds, dir_cls_preds = pred

        return super().loss(cls_scores, bbox_preds, dir_cls_preds, gt_bboxes_list, gt_labels_list, img_metas,
                            gt_bboxes_ignore)

    @force_fp32(apply_to=('pred'))
    def get_bboxes(self, pred, img_metas, rescale=False):
        cls_scores, bbox_preds, dir_cls_preds = pred
        return super().get_bboxes(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            img_metas,
            cfg=None,
            rescale=rescale)