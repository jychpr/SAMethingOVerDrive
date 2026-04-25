from util import box_ops
import torchvision
import torch
import torch.nn.functional as F
from .attention import multi_head_attention_forward_trans as MHA_woproj
import math
COCO_INDEX = [
    4,
    5,
    11,
    12,
    15,
    16,
    21,
    23,
    27,
    29,
    32,
    34,
    45,
    47,
    54,
    58,
    63,
]
LVIS_INDEX = [
    12,
    13,
    16,
    19,
    20,
    29,
    30,
    37,
    38,
    39,
    41,
    48,
    50,
    51,
    62,
    68,
    70,
    77,
    81,
    84,
    92,
    104,
    105,
    112,
    116,
    118,
    122,
    125,
    129,
    130,
    135,
    139,
    141,
    143,
    146,
    150,
    154,
    158,
    160,
    163,
    166,
    171,
    178,
    181,
    195,
    201,
    208,
    209,
    213,
    214,
    221,
    222,
    230,
    232,
    233,
    235,
    236,
    237,
    239,
    243,
    244,
    246,
    249,
    250,
    256,
    257,
    261,
    264,
    265,
    268,
    269,
    274,
    280,
    281,
    286,
    290,
    291,
    293,
    294,
    299,
    300,
    301,
    303,
    306,
    309,
    312,
    315,
    316,
    320,
    322,
    325,
    330,
    332,
    347,
    348,
    351,
    352,
    353,
    354,
    356,
    361,
    363,
    364,
    365,
    367,
    373,
    375,
    380,
    381,
    387,
    388,
    396,
    397,
    399,
    404,
    406,
    409,
    412,
    413,
    415,
    419,
    425,
    426,
    427,
    430,
    431,
    434,
    438,
    445,
    448,
    455,
    457,
    466,
    477,
    478,
    479,
    480,
    481,
    485,
    487,
    490,
    491,
    502,
    505,
    507,
    508,
    512,
    515,
    517,
    526,
    531,
    534,
    537,
    540,
    541,
    542,
    544,
    550,
    556,
    559,
    560,
    566,
    567,
    570,
    571,
    573,
    574,
    576,
    579,
    581,
    582,
    584,
    593,
    596,
    598,
    601,
    602,
    605,
    609,
    615,
    617,
    618,
    619,
    624,
    631,
    633,
    634,
    637,
    639,
    645,
    647,
    650,
    656,
    661,
    662,
    663,
    664,
    670,
    671,
    673,
    677,
    685,
    687,
    689,
    690,
    692,
    701,
    709,
    711,
    713,
    721,
    726,
    728,
    729,
    732,
    742,
    751,
    753,
    754,
    757,
    758,
    763,
    768,
    771,
    777,
    778,
    782,
    783,
    784,
    786,
    787,
    791,
    795,
    802,
    804,
    807,
    808,
    809,
    811,
    814,
    819,
    821,
    822,
    823,
    828,
    830,
    848,
    849,
    850,
    851,
    852,
    854,
    855,
    857,
    858,
    861,
    863,
    868,
    872,
    882,
    885,
    886,
    889,
    890,
    891,
    893,
    901,
    904,
    907,
    912,
    913,
    916,
    917,
    919,
    924,
    930,
    936,
    937,
    938,
    940,
    941,
    943,
    944,
    951,
    955,
    957,
    968,
    971,
    973,
    974,
    982,
    984,
    986,
    989,
    990,
    991,
    993,
    997,
    1002,
    1004,
    1009,
    1011,
    1014,
    1015,
    1027,
    1028,
    1029,
    1030,
    1031,
    1046,
    1047,
    1048,
    1052,
    1053,
    1056,
    1057,
    1074,
    1079,
    1083,
    1115,
    1117,
    1118,
    1123,
    1125,
    1128,
    1134,
    1143,
    1144,
    1145,
    1147,
    1149,
    1156,
    1157,
    1158,
    1164,
    1166,
    1192,
]

@torch.no_grad()
def sample_feature_vit(
    sizes,
    pred_boxes,
    features,
    unflatten=True,
):
    rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
    for i in range(len(rpn_boxes)):
        rpn_boxes[i][:, [0, 2]] = rpn_boxes[i][:, [0, 2]] * sizes[i][0]
        rpn_boxes[i][:, [1, 3]] = rpn_boxes[i][:, [1, 3]] * sizes[i][1]
    roi_features = torchvision.ops.roi_align(
        features,
        rpn_boxes,
        output_size=(1, 1),
        spatial_scale=1.0,
        aligned=True,
    )[..., 0, 0]
    normalized_roi_features =  F.normalize(roi_features, dim=-1, p=2)
    if unflatten:
        normalized_roi_features = normalized_roi_features.unflatten(0, (features.size(0), -1))
    return normalized_roi_features

@torch.no_grad()
def crop_and_resize_masks(sam_masks, rpn_boxes, grid_size, image_sizes):
    """Crop full-image 28x28 SAM masks to each RoI box region and resize to grid_size.

    sam_masks:   list of (N_i, 28, 28) bool tensors, one per image
    rpn_boxes:   list of (N_i, 4) float tensors in pixel xyxy per image
    image_sizes: list of (W, H) tuples per image
    grid_size:   int, target spatial size (7 for RN50, 9 for RN50x4)

    Returns: float tensor of shape (sum(N_i), 1, grid_size, grid_size)
    """
    results = []
    for masks_i, boxes_i, (W, H) in zip(sam_masks, rpn_boxes, image_sizes):
        n = boxes_i.shape[0]
        device = boxes_i.device
        # sizes values are 0-dim tensors; convert to plain float for math ops
        W_f, H_f = float(W), float(H)
        if masks_i is None or len(masks_i) == 0:
            results.append(torch.ones(n, 1, grid_size, grid_size, device=device))
            continue
        # masks_i: (N_i, 28, 28) bool on whatever device it came from
        masks_f = masks_i.to(device=device, dtype=torch.float32)  # (N_i, 28, 28)
        n_masks = masks_f.shape[0]
        cropped = []
        for j in range(n):
            x1, y1, x2, y2 = boxes_i[j].tolist()
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(W_f, x2); y2 = min(H_f, y2)
            # Map pixel box to 28x28 mask coords
            mx1 = x1 / W_f * 28; my1 = y1 / H_f * 28
            mx2 = x2 / W_f * 28; my2 = y2 / H_f * 28
            # Integer crop bounds
            c0 = max(0, math.floor(mx1)); r0 = max(0, math.floor(my1))
            c1 = min(28, math.ceil(mx2)); r1 = min(28, math.ceil(my2))
            if j < n_masks and r1 > r0 and c1 > c0:
                patch = masks_f[j, r0:r1, c0:c1].unsqueeze(0).unsqueeze(0)  # (1,1,rH,rW)
                resized = F.adaptive_max_pool2d(patch, (grid_size, grid_size))
            else:
                # No matching SAM mask or degenerate crop → neutral (no suppression)
                resized = torch.ones(1, 1, grid_size, grid_size, device=device)
            cropped.append(resized)
        results.append(torch.cat(cropped, dim=0))  # (n, 1, grid_size, grid_size)
    return torch.cat(results, dim=0)  # (Total_N, 1, grid_size, grid_size)


@torch.no_grad()
def sample_feature_rn(
    sizes,
    pred_boxes,
    features,
    args,
    backbone,
    extra_conv=False,
    unflatten=True,
    sam_masks=None,
):
    rpn_boxes = [box_ops.box_cxcywh_to_xyxy(pred) for pred in pred_boxes]
    for i in range(len(rpn_boxes)):
        rpn_boxes[i][:, [0, 2]] = rpn_boxes[i][:, [0, 2]] * sizes[i][0]
        rpn_boxes[i][:, [1, 3]] = rpn_boxes[i][:, [1, 3]] * sizes[i][1]
    if "RN50x4" in args.backbone:
        reso = 18
    else: 
        reso = 14
    if extra_conv:
        roi_features = torchvision.ops.roi_align(
            features,
            rpn_boxes,
            output_size=(reso, reso),
            spatial_scale=1.0,
            aligned=True,
        )
        if sam_masks is not None:
            mask_weights = crop_and_resize_masks(sam_masks, rpn_boxes, reso, sizes)
            roi_features = roi_features * mask_weights
        roi_features = backbone[0].layer4(roi_features)
        roi_features = backbone[0].attn_pool(roi_features, None)
    else:
        features = features.permute(0, 2, 3, 1)
        attn_pool = backbone[0].attn_pool
        q_feat = attn_pool.q_proj(features)
        k_feat = attn_pool.k_proj(features)
        v_feat = attn_pool.v_proj(features)
        hacked = False
        positional_emb = attn_pool.positional_embedding
        if not hacked:
            q_pe = F.linear(positional_emb[:1], attn_pool.q_proj.weight)
            k_pe = F.linear(positional_emb[1:], attn_pool.k_proj.weight)
            v_pe = F.linear(positional_emb[1:], attn_pool.v_proj.weight)
        if q_pe.dim() == 3:
            assert q_pe.size(0) == 1
            q_pe = q_pe[0]
        q, k, v = (
            q_feat.permute(0, 3, 1, 2),
            k_feat.permute(0, 3, 1, 2),
            v_feat.permute(0, 3, 1, 2),
        )
        q = torchvision.ops.roi_align(
            q,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )
        k = torchvision.ops.roi_align(
            k,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )
        v = torchvision.ops.roi_align(
            v,
            rpn_boxes,
            output_size=(reso // 2, reso // 2),
            spatial_scale=1.0,
            aligned=True,
        )

        # Triple-Filter: suppress background contamination via FastSAM binary masks.
        # When sam_masks is None this block is skipped entirely — pixel-identical fallback.
        if sam_masks is not None:
            grid_size = reso // 2
            mask_weights = crop_and_resize_masks(sam_masks, rpn_boxes, grid_size, sizes)
            q = q * mask_weights
            k = k * mask_weights
            v = v * mask_weights

        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        q = q.mean(-1)  # NC
        q = q + q_pe  # NC
        if k_pe.dim() == 3:
            k = k + k_pe.permute(1, 2, 0)
            v = v + v_pe.permute(1, 2, 0)
        else:
            k = k + k_pe.permute(1, 0).unsqueeze(0).contiguous()  # NC(HW)
            v = v + v_pe.permute(1, 0).unsqueeze(0).contiguous()  # NC(HW)
        q = q.unsqueeze(-1)
        roi_features = MHA_woproj(
            q,
            k,
            v,
            k.size(-2),
            attn_pool.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=attn_pool.c_proj.weight,
            out_proj_bias=attn_pool.c_proj.bias,
            training=False,
            out_dim=k.size(-2),
            need_weights=False,
        )[0][0]
        roi_features = roi_features.float()
    roi_features = roi_features / roi_features.norm(dim=-1, keepdim=True)
    if unflatten:
        roi_features = roi_features.unflatten(0, (features.size(0), -1))

    return roi_features

@torch.no_grad()
def get_match_pseudo_idx(query,targets,thr=0.5):
    bs,nq=query.shape[:2]
    device=query.device
    pseudo_gt_box=[]
    sizes=[]
    mask=[]
    proposal=query.clone()
    proposal=box_ops.box_cxcywh_to_xyxy(proposal.flatten(0, 1))
    for t in targets:
        if t.get('sam_boxes') is not None:
            # SAM path: use FastSAM boxes as reference geometry for wildcard IoU assignment
            bbox = box_ops.box_cxcywh_to_xyxy(t['sam_boxes'])
        else:
            # Legacy OLN path: use pseudo-labeled GT boxes
            pseudo_mask = t['pseudo_mask'].to(torch.bool)
            bbox = box_ops.box_cxcywh_to_xyxy(t['boxes'][pseudo_mask])
        pseudo_gt_box.append(bbox)
        sizes.append(len(bbox))
    pseudo_gt_box = torch.cat(pseudo_gt_box,dim=0)
    ious = box_ops.box_iou(proposal,pseudo_gt_box)[0]
    ious=ious.view(bs,nq,-1)
    for i, iou in enumerate(ious.split(sizes, -1)):
        if iou.numel()>0:
            bs_i_iou=iou[i]
            bs_i_miou=bs_i_iou.max(-1)[0]
            mask.append(bs_i_miou>thr)
        else:
            mask.append(torch.zeros(nq,dtype=torch.bool,device=device))
    return torch.stack(mask,dim=0)


@torch.no_grad()
def get_morph_labels_for_proposals(query, targets, thr=0.5):
    """Return (BS, NQ) long tensor of morph labels (0-3) for each proposal.

    For each proposal, finds the SAM detection with the highest IoU above `thr`
    and returns its morph_label.  Defaults to 0 (global wildcard) when no match
    is found or when sam_morph_labels is absent from a target.
    """
    bs, nq = query.shape[:2]
    device = query.device
    morph_out = torch.zeros(bs, nq, dtype=torch.long, device=device)

    sam_boxes_list = []
    morph_labels_list = []
    sizes = []
    for t in targets:
        if t.get('sam_boxes') is not None and t.get('sam_morph_labels') is not None:
            bbox = box_ops.box_cxcywh_to_xyxy(t['sam_boxes'])
            sam_boxes_list.append(bbox)
            morph_labels_list.append(t['sam_morph_labels'].to(device))
            sizes.append(len(bbox))
        else:
            sam_boxes_list.append(torch.zeros(0, 4, device=device))
            morph_labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
            sizes.append(0)

    if sum(sizes) == 0:
        return morph_out

    proposal = box_ops.box_cxcywh_to_xyxy(query.flatten(0, 1))
    all_sam_boxes = torch.cat(sam_boxes_list, dim=0)
    ious = box_ops.box_iou(proposal, all_sam_boxes)[0]  # (BS*NQ, total_sam)
    ious = ious.view(bs, nq, -1)                         # (BS, NQ, total_sam)

    for i, (iou_chunk, morph_i) in enumerate(zip(ious.split(sizes, -1), morph_labels_list)):
        # iou_chunk: (BS, NQ, n_sam_i) — only column-set i belongs to image i
        if iou_chunk.size(-1) == 0:
            continue
        iou_i = iou_chunk[i]                           # (NQ, n_sam_i)
        best_iou, best_idx = iou_i.max(-1)             # (NQ,)
        matched = best_iou > thr
        if matched.any():
            best_morph = morph_i[best_idx]             # (NQ,) — may index out-of-range if
            morph_out[i, matched] = best_morph[matched]  # morph_i is non-empty (guaranteed by sizes>0)

    return morph_out


@torch.no_grad()
def get_matched_soft_wildcards(query, targets, thr=0.5):
    """Return (BS, NQ, text_dim) soft wildcard embeddings matched by IoU.

    Zero vectors are returned for proposals with no IoU match above thr.
    Caller checks norm > 0 to distinguish matched from unmatched proposals.
    """
    bs, nq = query.shape[:2]
    device = query.device

    sam_boxes_list = []
    sw_list = []
    sizes = []
    text_dim = 0

    for t in targets:
        if t.get('sam_boxes') is not None and t.get('sam_soft_wildcards') is not None:
            bbox = box_ops.box_cxcywh_to_xyxy(t['sam_boxes'])
            sw = t['sam_soft_wildcards'].to(device)
            sam_boxes_list.append(bbox)
            sw_list.append(sw)
            sizes.append(len(bbox))
            text_dim = sw.shape[-1]
        else:
            sam_boxes_list.append(torch.zeros(0, 4, device=device))
            sw_list.append(None)
            sizes.append(0)

    if text_dim == 0 or sum(sizes) == 0:
        return torch.zeros(bs, nq, max(text_dim, 1), device=device)

    out = torch.zeros(bs, nq, text_dim, device=device)
    proposal = box_ops.box_cxcywh_to_xyxy(query.flatten(0, 1))
    all_sam_boxes = torch.cat(sam_boxes_list, dim=0)
    ious = box_ops.box_iou(proposal, all_sam_boxes)[0]  # (BS*NQ, total_sam)
    ious = ious.view(bs, nq, -1)

    for i, (iou_chunk, sw_i) in enumerate(zip(ious.split(sizes, -1), sw_list)):
        if iou_chunk.size(-1) == 0 or sw_i is None:
            continue
        iou_i = iou_chunk[i]
        best_iou, best_idx = iou_i.max(-1)
        matched = best_iou > thr
        if matched.any():
            out[i, matched] = sw_i[best_idx[matched]]

    return out
