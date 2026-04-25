import torch
from torchvision.ops import box_iou as tv_box_iou
from util import box_ops


@torch.no_grad()
def mine_disagreement_regions(
    pred_logits,
    pred_boxes,
    sam_boxes,
    sam_scores,
    base_class_count,
    entropy_threshold=1.5,
    iou_threshold=0.3,
    weight_scale=0.15,
):
    """
    pred_logits:  (B, N, C) softmax distribution over base classes
    pred_boxes:   (B, N, 4) cxcywh normalized, detached
    sam_boxes:    list[B] of (M_i, 4) cxcywh tensors or None entries
    sam_scores:   list[B] of (M_i,) tensors or None (kept for API symmetry)
    Returns list[B] of dicts {'boxes': (K,4), 'scores': (K,)} or {} if no candidates.
    """
    p = pred_logits.clamp(min=1e-9)
    entropy = -(p * p.log()).sum(dim=-1)  # (B, N) Shannon entropy

    result = []
    for i in range(len(pred_logits)):
        high = entropy[i] > entropy_threshold  # (N,)
        if not high.any():
            result.append({})
            continue

        boxes = pred_boxes[i][high]  # (K, 4)
        ent   = entropy[i][high]     # (K,)

        sb = sam_boxes[i] if (sam_boxes is not None and i < len(sam_boxes)) else None
        if sb is not None and len(sb) > 0:
            b_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.to(sb.device))
            s_xyxy = box_ops.box_cxcywh_to_xyxy(sb)
            ious = tv_box_iou(b_xyxy, s_xyxy)  # (K, M)
            no_overlap = ious.max(dim=1).values <= iou_threshold
            boxes = boxes[no_overlap]
            ent   = ent[no_overlap]

        if len(boxes) == 0:
            result.append({})
            continue

        result.append({
            'boxes':  boxes,
            'scores': (ent * weight_scale).clamp(max=0.3),
        })

    return result
