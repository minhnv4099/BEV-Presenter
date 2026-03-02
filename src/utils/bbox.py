#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape (N, 5).
        labels (torch.Tensor): Labels with shape (N, ).
        scores (torch.Tensor): Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        bboxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attr_labels'] = attrs.cpu()

    return result_dict


def box3d_multiclass_nms(boxes3d, boxes_for_nms, scores, score_thr, max_per_frame, nms_cfg, mlvl_attr_scores=None):
    """
    boxes3d: (N, 7) - predicted boxes
    boxes_for_nms: (N, 7) - boxes projected for NMS (BEV)
    scores: (N,) - confidence scores
    score_thr: float - min score to keep
    max_per_frame: int - max boxes to keep per frame
    mlvl_attr_scores: (N, K) optional attribute scores
    """

    # 1️⃣ filter by score threshold
    keep = scores >= score_thr
    boxes3d = boxes3d[keep]
    boxes_for_nms = boxes_for_nms[keep]
    scores = scores[keep]
    if mlvl_attr_scores is not None:
        mlvl_attr_scores = mlvl_attr_scores[keep]

    # 2️⃣ simple BEV IoU computation for NMS
    # BEV: use x, y, w, l
    # note: for demo, we use a simple IoU function for rectangles
    def iou_bev(box_a, box_b):
        # box = (x, y, w, l)
        xa, ya, wa, la = box_a
        xb, yb, wb, lb = box_b
        # compute intersection
        x1 = max(xa - wa/2, xb - wb/2)
        y1 = max(ya - la/2, yb - lb/2)
        x2 = min(xa + wa/2, xb + wb/2)
        y2 = min(ya + la/2, yb + lb/2)
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = wa * la
        area_b = wb * lb
        return inter_area / (area_a + area_b - inter_area + 1e-6)

    # 3️⃣ sort by scores descending
    order = scores.argsort(descending=True)
    boxes3d = boxes3d[order]
    boxes_for_nms = boxes_for_nms[order]
    scores = scores[order]
    if mlvl_attr_scores is not None:
        mlvl_attr_scores = mlvl_attr_scores[order]

    keep_idx = []
    for i in range(len(boxes_for_nms)):
        keep_flag = True
        for j in keep_idx:
            if iou_bev(boxes_for_nms[i][:4], boxes_for_nms[j][:4]) > nms_cfg.get("iou_thr", 0.01):
                keep_flag = False
                break
        if keep_flag:
            keep_idx.append(i)
        if len(keep_idx) >= max_per_frame:
            break

    boxes3d = boxes3d[keep_idx]
    scores = scores[keep_idx]
    labels = torch.zeros(len(keep_idx), dtype=torch.int64)  # giả định class 0 nếu multiclass chưa implement
    if mlvl_attr_scores is not None:
        attrs = mlvl_attr_scores[keep_idx]
    else:
        attrs = None

    return boxes3d, scores, labels, attrs
