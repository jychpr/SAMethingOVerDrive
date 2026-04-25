"""
Offline FastSAM prior precomputation for COCO train2017 and val2017.

Saves per-image .pt files to data/sam_priors/{split}/{image_id}.pt
Each file contains:
  boxes  : Nx4 float32  (cx, cy, w, h normalized)
  masks  : Nx28x28 bool (binary masks downsampled to 28x28)
  scores : N    float32 (detection confidence)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import FastSAM

TOP_K = 30
MIN_AREA = 0.0005
MASK_SIZE = 28

_DUMMY = {
    "boxes":       torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    "masks":       torch.zeros(1, MASK_SIZE, MASK_SIZE, dtype=torch.bool),
    "scores":      torch.zeros(1),
    "morph_labels": torch.zeros(1, dtype=torch.int64),
}


def _xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return torch.stack([cx, cy, bw, bh], dim=1)


def _resize_masks(masks: torch.Tensor) -> torch.Tensor:
    # masks: Nx H x W float
    out = F.interpolate(
        masks.unsqueeze(1), size=(MASK_SIZE, MASK_SIZE), mode="nearest"
    )
    return out.squeeze(1).bool()


def _compute_morph_label(mask: torch.Tensor, box_cxcywh_norm: torch.Tensor) -> int:
    """Assign a shape-conditioned morphology bucket label (0-3) from a 28x28 bool mask.

    Labels:
      0  default / fallback ("a photo of an object")
      1  small   normalized_area < 0.005
      2  large   normalized_area > 0.15
      3  thin    elongation > 3.5 or aspect_ratio > 3.0 or aspect_ratio < 0.33

    Priority: check labels in order 1→2→3; first match wins.
    """
    # normalized_area from bbox (more reliable than counting 28x28 pixels)
    w_n, h_n = box_cxcywh_norm[2].item(), box_cxcywh_norm[3].item()
    normalized_area = w_n * h_n

    mask_np = mask.numpy().astype(bool)

    # aspect_ratio from 28x28 bounding rect of non-zero pixels
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any() or not cols.any():
        return 0
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    bbox_h = max(int(r1 - r0) + 1, 1)
    bbox_w = max(int(c1 - c0) + 1, 1)
    aspect_ratio = bbox_w / bbox_h

    # elongation via PCA on foreground pixel coordinates
    ys, xs = np.where(mask_np)
    if len(ys) < 2:
        elongation = 1.0
    else:
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        coords -= coords.mean(axis=0)
        cov = (coords.T @ coords) / max(len(coords) - 1, 1)
        a, b, c = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
        disc = max(((a - c) / 2) ** 2 + b * b, 0.0)
        lam1 = (a + c) / 2 + disc ** 0.5
        lam2 = max((a + c) / 2 - disc ** 0.5, 1e-6)
        elongation = (lam1 / lam2) ** 0.5

    if normalized_area < 0.005:
        return 1
    if normalized_area > 0.15:
        return 2
    if elongation > 3.5 or aspect_ratio > 3.0 or aspect_ratio < 0.33:
        return 3
    return 0


def process_split(model: FastSAM, image_dir: Path, ann_path: Path, out_dir: Path) -> None:
    with open(ann_path) as f:
        images = json.load(f)["images"]

    out_dir.mkdir(parents=True, exist_ok=True)

    for img_info in tqdm(images, desc=ann_path.stem):
        image_id = img_info["id"]
        out_path = out_dir / f"{image_id}.pt"
        if out_path.exists():
            continue

        img_path = image_dir / img_info["file_name"]
        results = model(
            str(img_path),
            conf=0.3,
            iou=0.6,
            imgsz=1024,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            torch.save(_DUMMY, out_path)
            continue

        h, w = result.orig_shape
        boxes_norm = _xyxy_to_cxcywh_norm(result.boxes.xyxy.cpu(), w, h)
        scores = result.boxes.conf.cpu()

        # drop sub-pixel / texture fragments
        area = boxes_norm[:, 2] * boxes_norm[:, 3]
        keep = area >= MIN_AREA
        boxes_norm = boxes_norm[keep]
        scores = scores[keep]

        if len(scores) == 0:
            torch.save(_DUMMY, out_path)
            continue

        # sort by confidence descending, keep top-K
        order = scores.argsort(descending=True)[:TOP_K]
        boxes_norm = boxes_norm[order]
        scores = scores[order]

        if result.masks is not None:
            masks_raw = result.masks.data.cpu()[keep][order].float()
            masks = _resize_masks(masks_raw)
        else:
            masks = torch.zeros(len(scores), MASK_SIZE, MASK_SIZE, dtype=torch.bool)

        morph_labels = torch.tensor(
            [_compute_morph_label(masks[j], boxes_norm[j]) for j in range(len(boxes_norm))],
            dtype=torch.int64,
        )
        torch.save(
            {"boxes": boxes_norm, "masks": masks, "scores": scores, "morph_labels": morph_labels},
            out_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute FastSAM priors for COCO")
    parser.add_argument("--model", default="FastSAM-x.pt", help="FastSAM weights path")
    parser.add_argument("--data_root", default="data", help="Root data directory")
    parser.add_argument(
        "--splits", nargs="+", default=["train2017", "val2017"], help="Splits to process"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model = FastSAM(args.model)

    for split in args.splits:
        ann_path = data_root / "Annotations" / f"instances_{split}.json"
        image_dir = data_root / "Images" / split
        out_dir = data_root / "sam_priors" / split
        process_split(model, image_dir, ann_path, out_dir)


if __name__ == "__main__":
    main()
