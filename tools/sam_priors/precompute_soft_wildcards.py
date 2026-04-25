"""
Precompute CLIP-retrieved soft wildcard embeddings for FastSAM proposals.

For each SAM proposal in each .pt file, extracts a mask-purified CLIP visual embedding,
retrieves top-3 vocabulary entries by cosine similarity, and saves the centroid of their
text embeddings as 'soft_wildcard_embs' (N, text_dim) float32.
"""

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.clip.clip import load as clip_load

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CROP_SIZE = 224
TOP_K = 3


def make_masked_crop(img_t, box_cxcywh_norm, mask28):
    """Return (3, 224, 224) CLIP-normalized crop with background masked out.

    img_t: (3, H, W) float32 in [0, 1]
    box_cxcywh_norm: (4,) cx,cy,w,h normalized to [0, 1]
    mask28: (28, 28) bool, full-image mask downsampled to 28x28
    """
    _, H, W = img_t.shape
    cx, cy, bw, bh = box_cxcywh_norm.tolist()
    x1 = max(0, int((cx - bw / 2) * W))
    y1 = max(0, int((cy - bh / 2) * H))
    x2 = min(W, int((cx + bw / 2) * W))
    y2 = min(H, int((cy + bh / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return (torch.zeros(3, CROP_SIZE, CROP_SIZE) - CLIP_MEAN) / CLIP_STD

    crop = img_t[:, y1:y2, x1:x2].clone()
    ch, cw = y2 - y1, x2 - x1

    # Crop the 28x28 mask to box region and resize to crop resolution
    mx1 = max(0, math.floor(x1 / W * 28))
    my1 = max(0, math.floor(y1 / H * 28))
    mx2 = min(28, math.ceil(x2 / W * 28))
    my2 = min(28, math.ceil(y2 / H * 28))
    if mx2 > mx1 and my2 > my1:
        patch = mask28[my1:my2, mx1:mx2].float().unsqueeze(0).unsqueeze(0)
        mask_c = F.interpolate(patch, size=(ch, cw), mode='nearest')[0, 0]
        crop = crop * mask_c.unsqueeze(0)

    crop = F.interpolate(crop.unsqueeze(0), size=(CROP_SIZE, CROP_SIZE),
                         mode='bilinear', align_corners=False).squeeze(0)
    return (crop - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def process_image(prior_path, img_path, vocab_embs, clip_model, device, batch_size):
    sam_data = torch.load(prior_path, weights_only=True)
    if "soft_wildcard_embs" in sam_data:
        return

    boxes = sam_data["boxes"]   # (N, 4) cxcywh norm
    masks = sam_data["masks"]   # (N, 28, 28) bool
    N = len(boxes)
    if N == 0:
        return

    pil_img = Image.open(img_path).convert("RGB")
    img_t = TF.to_tensor(pil_img)  # (3, H, W) float32 [0, 1]

    crops = [make_masked_crop(img_t, boxes[j], masks[j]) for j in range(N)]

    vis_embs_list = []
    for i in range(0, N, batch_size):
        batch = torch.stack(crops[i:i + batch_size]).to(device)
        embs = clip_model.encode_image(batch).float()
        vis_embs_list.append(F.normalize(embs, dim=-1).cpu())
    vis_embs = torch.cat(vis_embs_list, dim=0)  # (N, D)

    sims = vis_embs @ vocab_embs.t()              # (N, M)
    top3_idx = sims.topk(TOP_K, dim=-1).indices   # (N, 3)
    soft_wildcards = vocab_embs[top3_idx].mean(dim=1)  # (N, D)
    soft_wildcards = F.normalize(soft_wildcards, dim=-1).float()

    sam_data["soft_wildcard_embs"] = soft_wildcards
    torch.save(sam_data, prior_path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute soft wildcard embeddings for FastSAM proposals"
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--vocab", default="pretrained/vocab_embeddings.pt",
                        help="Vocab embeddings from build_vocab_embeddings.py")
    parser.add_argument("--splits", nargs="+", default=["train2017"],
                        help="Splits to process (must have .pt files in sam_priors/)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vocab_data = torch.load(args.vocab, weights_only=True)
    vocab_embs = vocab_data["embeddings"]  # (M, D) normalized, on CPU

    clip_model, _ = clip_load("RN50", device=args.device)
    clip_model.eval()

    data_root = Path(args.data_root)

    for split in args.splits:
        image_dir = data_root / "Images" / split
        prior_dir = data_root / "sam_priors" / split
        if not prior_dir.exists():
            print(f"Skipping {split}: {prior_dir} not found")
            continue

        prior_paths = sorted(prior_dir.glob("*.pt"))
        for prior_path in tqdm(prior_paths, desc=split):
            image_id = int(prior_path.stem)
            img_path = image_dir / f"{image_id:012d}.jpg"
            if not img_path.exists():
                continue
            process_image(prior_path, img_path, vocab_embs, clip_model, args.device, args.batch_size)


if __name__ == "__main__":
    main()
