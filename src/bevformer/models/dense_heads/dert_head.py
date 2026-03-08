#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import copy
import torch
import torch.nn as nn
from typing import Optional, Sequence

from mmengine.model import BaseModule
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import FFN

from src.typing_ import ConfigType
from src.utils.logging import getLogger
from src.registry import HEADS, BBOX_SAMPLERS
from src.bevformer.models.utils.bricks import build_transformer

from src.bevformer.core.bbox import build_assigner
from src.bevformer.builder import build_loss

logger = getLogger(__name__)


@HEADS.register_module()
class DETRHead(BaseModule):
    r"""Head of DETR. DETR:End-to-End Object Detection with Transformers.

    More details can be found in the `paper
    <https://arxiv.org/pdf/2005.12872>`_ .

    Args:
        num_classes (int): Number of categories excluding the background.
        embed_dims (int): The dims of Transformer embedding.
        num_reg_fcs (int): Number of fully-connected layers used in `FFN`,
            which is then used for the regression head. Defaults to `2`.
        sync_cls_avg_factor (bool): Whether to sync the `avg_factor` of
            all ranks. Default to `False`.
        transformer (ConfigType): Config of the transformer module. Default to ``.
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_bbox (:obj:`ConfigDict` or dict): Config of the regression bbox
            loss. Defaults to `L1Loss`.
        loss_iou (:obj:`ConfigDict` or dict): Config of the regression iou
            loss. Defaults to `GIoULoss`.
        train_cfg (:obj:`ConfigDict` or dict): Training config of transformer head.
        test_cfg (:obj:`ConfigDict` or dict): Testing config of transformer head.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    _version = 2

    def __init__(
        self,
        num_classes: int,
        embed_dims: int = 256,
        num_reg_fcs: int = 2,
        sync_cls_avg_factor: bool = False,
        transformer: Optional[ConfigType] = None,
        loss_cls: ConfigType = dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
        loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
        train_cfg: ConfigType = dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=1.),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
        test_cfg: ConfigType = dict(max_per_img=100),
        init_cfg: ConfigType = None,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), \
                f'Expected class_weight to have type float. Found {type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            if train_cfg.get('sampler', None) is not None:
                logger.warning('DETR do not build sampler.')

            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = BBOX_SAMPLERS.build(sampler_cfg)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        self.transformer = build_transformer(transformer)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        # cls branch
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.activate = nn.ReLU()
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            dict(type='ReLU', inplace=True),
            dropout=0.0,
            add_residual=False)
        # NOTE the activations of reg_branch here is the same as
        # those in transformer, but they are actually different
        # in DAB-DETR (prelu in transformer and relu in reg_branch)
        self.fc_reg = Linear(self.embed_dims, 4)

    def forward(self):
        raise NotImplementedError
