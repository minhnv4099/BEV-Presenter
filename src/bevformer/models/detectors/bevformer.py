# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Minh Nguyen
# ---------------------------------------------
from __future__ import annotations

import copy
from typing import Optional, TYPE_CHECKING, Literal, Union, Dict
import torch

from src.registry import DETECTORS
from src.bevformer.models.utils.grid_mask import GridMask
from src.bevformer.models.utils.bbox import bbox3d2result
from src.utils.logging import getLogger
from src.utils.fp16_utils import auto_fp16
from .mvx_two_stage import MVXTwoStageDetector

if TYPE_CHECKING:
    from src.modeling_output import BackboneOutput
    from src.structures.bbox_3d import BaseInstance3DBoxes

logger = getLogger(__name__)


@DETECTORS.register_module()
class BEVFormerDetector(MVXTwoStageDetector):
    """BEVFormer Detector, an end-to-end detector.

    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask: bool = False,
        pretrained: Optional[str] = None,
        video_test_mode: bool = False,
        pts_voxel_layer: Optional[Dict] = None,
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
        MVXTwoStageDetector.__init__(
            self,
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
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
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

    def forward(
        self,
        *args,
        return_loss: bool = True,
        mode: Literal['loss', 'predict', 'val'] = 'loss',
        **kwargs
    ):
        """Calls either forward_train or forward_test depending on whether
        `return_loss=True`.

        Note::

        This setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `return_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if mode == 'loss':
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        points: Optional[list[torch.Tensor]] = None,
        gt_bboxes_3d: Optional[list['BaseInstance3DBoxes']] = None,
        gt_labels_3d: Optional[list[torch.Tensor]] = None,
        gt_labels: Optional[list[torch.Tensor]] = None,
        gt_bboxes: Optional[list[torch.Tensor]] = None,
        gt_bboxes_ignore: Optional[list[torch.Tensor]] = None,
        proposals: Optional[list[torch.Tensor]] = None
    ):
        """Forward training function.

        Args:
            img (torch.Tensor): Images of each sample with shape
                `(bs, n_queue, num_cam, C, H, W)`.
                The last `n_queue - 1` are previous frames.
            img_metas (list[dict], optional): Meta information of each sample.
                List of `bs` dict of `n_queue`.
            points (list[torch.Tensor], optional): Points of each sample.
                Length of `bs`. Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. List of 'bs'. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. List if 'bs'. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. List if 'bs'. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. List if 'bs'. Defaults to None.
            proposals (list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        curr_img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # last meta, i.e. img meta for current timestamp
        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # list of `n_levels` of `(bs, n_cam, c, h, w)`
        img_feats = self.extract_feat(pixel_values=curr_img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            prev_bev=prev_bev
        )

        losses.update(losses_pts)
        return losses

    def obtain_history_bev(self, imgs_queue: torch.Tensor, img_metas_list: list[dict]):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        Args:
            imgs_queue (torch.Tensor):
                Queue of previous images.
                Shape of `(bs, n_queue - 1, n_cam, C, H, W)`.
            img_metas_list (list[dict]):
                List of metadata of images in queue. Length of 'bs', each has length of `n_queue`.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            # list of `n_levels` of `(bs, len_queue, n_cam, C, H, W)`
            img_feats_list = self.extract_img_feat(
                pixel_values=imgs_queue,
                len_queue=len_queue,
                img_metas=img_metas_list)

            # over queue to get prev_bev
            for i in range(len_queue):
                img_metas = [each_sample[i] for each_sample in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None

                # get each element in queue
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                prev_bev = self.pts_bbox_head(
                    mlvl_feats=img_feats,
                    prev_bev=prev_bev,
                    img_metas=img_metas,
                    only_bev=True)

            self.train()

            return prev_bev

    def extract_feat(self, pixel_values, img_metas: list[dict], len_queue: int = None) -> list[torch.Tensor]:
        """Extract feature maps of images and points via backbones and optionally neck (FPN).

        Args:
            pixel_values (`torch.Tensor`): Batch of images to extract features maps.
            img_metas (`list[dict]): Metadata of images.
            len_queue (`int`, *optional* default to ``None``): Queue length.
        Returns:
            Tuple of tensor with shape `[..., len_queue, num_cam, c, h, w]`.
        """
        img_feats = self.extract_img_feat(pixel_values, img_metas, len_queue=len_queue)
        return img_feats

    @auto_fp16(apply_to=('pixel_values',))
    def extract_img_feat(
        self,
        pixel_values: torch.Tensor,
        img_metas: list[dict],
        len_queue: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Extract feature maps of image via backbones and optionally neck (FPN).

        Args:
            pixel_values (`torch.FloatTensor`):
                Batch of images to extract features maps.
                Shape of `[bs, n_cam, C (3), H, W]`
            img_metas (`list[dict]):
                List of `bs` metadata of images.
            len_queue (`int`, *optional* default to ``None``): Queue length.
                Indicate by the first dimension (i.e. bs).
        Returns:
            tuple[Tensor]: Tuple of `n_levels` tensor with shape:

                - `[bs, len_queue, n_cam, c, h, w]` if ``len_queue`` is provided.
                - `[bs, n_cam, c, h, w]` if ``len_queue`` is `None`.
        """
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

        if self.use_grid_mask:
            pixel_values = self.grid_mask(pixel_values)

        feature_maps = self.img_backbone(pixel_values)

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

    def forward_pts_train(
        self,
        pts_feats: list[torch.Tensor],
        gt_bboxes_3d: list,
        gt_labels_3d: list[torch.Tensor],
        img_metas: list[dict],
        gt_bboxes_ignore: Optional[list[torch.Tensor]] = None,
        prev_bev: Optional[torch.Tensor] = None
    ):
        """Forward function to forward ``pts_bbox_head``.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample.
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

    def forward_test(
        self,
        img: list[torch.Tensor],
        img_metas: list[list[dict]],
        **kwargs
    ):
        """Forward testing

        Args:
            img (list[Tensor]):
                List of `n_aug` tensor with shape of `(bs, n_cam, c, h, w)`.
                Only use current timestamp, not queue of frames.
            img_metas (list[list[dict]]):
                List of `n_aug` lists of `bs`.

        Returns:
            list[dict]: List of `bs` dict with keys:

                - bboxes_3d: Tensor like shape `(n_query, 9)`.
                - scores_3d: Tensor like shape `(n_query, )`.
                - labels_3d: Tensor like shape `(n_query, )`.
        """
        if not isinstance(img_metas, list):
            raise TypeError(f"'img_metas' must be a list, but got {type(img_metas)}")
        if not isinstance(img, list):
            raise TypeError(f"'img' must be a list, but got {type(img)}")

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

        # test on only first augmentation
        new_prev_bev, bbox_results = self.simple_test(
            img[0], img_metas[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        pixel_values: torch.FloatTensor,
        img_metas: list[dict],
        prev_bev: Optional[torch.Tensor],
        rescale: bool = False
    ):
        """Simple test function without augmentation predict results of bev embeddings and bounding boxes.

          Args:
              pixel_values (`torch.FloatTensor`):
                  Feature maps for testing images.
                  Shape `(bs, n_cam, c, h, w)`.
              img_metas (`list[dict]'): Metadata of images.
                List of `bs`.
              prev_bev: Previous BEV embeddings at timestamp t-1.
              rescale (`bool`): Whether to rescale.

          Returns:
              2-element tuple: BEV embeddings and list of `bs` of predicted bboxes.
          """
        img_feats = self.extract_feat(pixel_values=pixel_values, img_metas=img_metas)

        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return outs['bev_embed'], bbox_results

    def simple_test_pts(
        self,
        feature_maps: list[torch.Tensor],
        img_metas,
        prev_bev=None,
        rescale=False
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
