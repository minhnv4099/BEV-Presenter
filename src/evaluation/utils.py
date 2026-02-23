#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional, Tuple
import torch
from torch import Tensor


def box3d_multiclass_nms(
        mlvl_bboxes: Tensor,
        mlvl_bboxes_for_nms: Tensor,
        mlvl_scores: Tensor,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores: Optional[Tensor] = None,
        mlvl_attr_scores: Optional[Tensor] = None,
        mlvl_bboxes2d: Optional[Tensor] = None) -> Tuple[Tensor]:
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (Tensor): Multi-level boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (Tensor): Multi-level boxes with shape (N, C + 1).
            N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (Tensor, optional): Multi-level scores of direction
            classifier. Defaults to None.
        mlvl_attr_scores (Tensor, optional): Multi-level scores of attribute
            classifier. Defaults to None.
        mlvl_bboxes2d (Tensor, optional): Multi-level 2D bounding boxes.
            Defaults to None.

    Returns:
        Tuple[Tensor]: Return results after nms, including 3D bounding boxes,
        scores, labels, direction scores, attribute scores (optional) and
        2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results