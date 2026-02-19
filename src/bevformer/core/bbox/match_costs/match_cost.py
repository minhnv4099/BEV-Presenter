# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Minh Nguyen
# ---------------------------------------------
import torch
import torch.nn as nn
from src.registry import MATCH_COST
from src.bevformer.models.utils.weighted import weighted_loss


@MATCH_COST.register_module()
class FocalCost(nn.Module):
    """
    Focal classification cost for Hungarian matching.

    Args:
        weight (float): cost weight.
        alpha (float): focal alpha.
        gamma (float): focal gamma.
        eps (float): numerical stability.
    """

    def __init__(self,
                 weight: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self,
                pred_logits: torch.Tensor,
                gt_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (num_queries, num_classes)
            gt_labels:   (num_gts,)  class index

        Returns:
            cost: (num_queries, num_gts)
        """
        # sigmoid probability
        pred_prob = pred_logits.sigmoid()  # (Q, C)
        pt = pred_prob[:, gt_labels.detach()]

        # focal positive cost
        pos_cost = - self.alpha * ((1 - pt) ** self.gamma) * torch.log(pt + self.eps)

        return pos_cost * self.weight


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss.sum(-1)


@MATCH_COST.register_module()
class SmoothL1Cost(object):
    """SmoothL1Cost.
     Args:
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        pred = bboxes.contiguous().view(bboxes.shape)[:, None, :]
        target = gt_bboxes.contiguous().view(gt_bboxes.shape)[None, :, :]
        cost = smooth_l1_loss(pred, target)

        return cost * self.weight
