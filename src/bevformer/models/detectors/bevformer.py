# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Minh Nguyen
# ---------------------------------------------
from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING
import torch

from src.registry import MODELS, DETECTORS
from src.bevformer.models.utils.grid_mask import GridMask
from src.bevformer.models.utils.bbox import bbox3d2result
from src.utils.logging import getLogger
from src.utils.fp16_utils import auto_fp16
from .mvx_two_stage import MVXTwoStageDetector

if TYPE_CHECKING:
    from src.modeling_output import BackboneOutput


logger = getLogger(__name__)


@MODELS.register_module()
@DETECTORS.register_module()
class BEVFormerDetector(MVXTwoStageDetector):
    """BEVFormer.

    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        # arguments for self
        use_grid_mask=False,
        pts_voxel_layer=None,
        pretrained=None,
        video_test_mode=False,
        # arguments for supper
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        pts_bbox_head: Optional[dict] = None,
        img_roi_head: Optional[dict] = None,
        img_rpn_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            **kwargs
        )
        self.grid_mask = GridMask(
            use_h=True, use_w=True,
            rotate=1, offset=False,
            ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    @auto_fp16(apply_to=('pixel_values',))
    def extract_img_feat(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        img_metas: Optional[list[dict]] = None,
        len_queue: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Extract feature maps of image via backbones and optionally neck (FPN).

        Args:
            pixel_values (`torch.FloatTensor`): Batch of images to extract features maps.
            img_metas (`list[dict], *optional* default to None): Metadata of images.
            len_queue (`int`, *optional* default to ``None``): Queue length.
        """
        if pixel_values is None:
            return None

        B = pixel_values.size(0)

        input_shape = pixel_values.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        # reshape img to 4-dim
        if pixel_values.dim() == 5 and B == 1:
            pixel_values.squeeze_()
        elif pixel_values.dim() == 5 and B > 1:
            B, N, C, H, W = pixel_values.size()
            pixel_values = pixel_values.reshape(B * N, C, H, W)
        # TODO: inspect grid_mask
        if self.use_grid_mask:
            pixel_values = self.grid_mask(pixel_values)

        backbone_output: "BackboneOutput" = self.img_backbone(pixel_values)
        feature_maps = backbone_output.feature_maps

        if self.with_img_neck:
            feature_maps = self.img_neck(feature_maps)

        img_feats_reshaped = []
        for feature_map in feature_maps:
            BN, C, H, W = feature_map.size()
            if len_queue is not None:
                img_feats_reshaped.append(feature_map.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(feature_map.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def extract_feat(self, pixel_values, img_metas=None, len_queue=None) -> list[torch.Tensor]:
        """Extract feature maps of images and points via backbones and optionally neck (FPN).

        Args:
            pixel_values (`torch.FloatTensor`): Batch of images to extract features maps.
            img_metas (`list[dict], *optional* default to None): Metadata of images.
            len_queue (`int`, *optional* default to ``None``): Queue length.
        """

        img_feats = self.extract_img_feat(pixel_values, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None
    ):
        """Forward function

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            mlvl_feats=pts_feats,
            img_metas=img_metas,
            prev_bev=prev_bev
        )
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note::

        This setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `return_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue: torch.Tensor, img_metas_list: list[dict]):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)

            self.train()

            return prev_bev

    def forward_train(
        self,
        points: Optional[list[torch.Tensor]] = None,
        img_metas: Optional[list[dict]] = None,
        gt_bboxes_3d: Optional[list['BaseInstance3DBoxes']] = None,
        gt_labels_3d: Optional[list[torch.Tensor]] = None,
        gt_labels: Optional[list[torch.Tensor]] = None,
        gt_bboxes: Optional[list[torch.Tensor]] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        proposals: Optional[list[torch.Tensor]] = None,
        gt_bboxes_ignore: Optional[list[torch.Tensor]] = None,
        img_depth=None,
        img_mask=None,
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            pixel_values (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals (list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = pixel_values.size(1)
        prev_img = pixel_values[:, :-1, ...]
        img = pixel_values[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(pixel_values=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            prev_bev=prev_bev
        )

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError(f"'img_metas' must be a list, but got {type(img_metas)}")

        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def simple_test(self, pixel_values: torch.FloatTensor, img_metas: list[dict], prev_bev=None, rescale=False):
        """Simple test function without augmentation predict results of bev embeddings and bounding boxes.

          Args:
              pixel_values (`torch.FloatTensor`):
                  Feature maps for testing images.
              img_metas (`list[dict]'): Metadata of images.
              prev_bev: Previous BEV embeddings at timestamp t-1.
              rescale (`bool`): Whether to rescale.

          Returns:
              2-element tuple: BEV embeddings and predicted bboxes.
          """
        img_feats = self.extract_feat(pixel_values=pixel_values, img_metas=img_metas)
        new_prev_bev, bbox_pts = self.simple_test_pts(
            feature_maps=img_feats,
            img_metas=img_metas,
            prev_bev=prev_bev,
            rescale=rescale
        )

        return new_prev_bev, bbox_pts

    def simple_test_pts(
        self, feature_maps: list[torch.Tensor],
        img_metas, prev_bev=None, rescale=False
    ) -> tuple[torch.Tensor, list[list]]:
        """Simple predict results of bev embeddings and bounding boxes.

        Args:
            feature_maps (`list[torch.Tensor]`):
                Feature maps for testing images.
            img_metas (`list[dict]'): Metadata of images.
            prev_bev: Previous BEV embeddings at timestamp t-1.
            rescale (`bool`): Whether to rescale.

        Returns:
            2-element tuple: BEV embeddings and predicted bboxes.
        """
        outs = self.pts_bbox_head(feature_maps, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return outs['bev_embed'], bbox_results
