# SAMethingOVerDrive
SAM Priors Localization for Open Vocabulary Detection

# Setup
1. Clone the repository and go to directory
```bash
git clone git@github.com:jychpr/SAMethingOVerDrive.git
cd SAMethingOVerDrive/
```
2. Create environment
```bash
conda create -n SAMOVD python=3.10 -y
conda activate SAMOVD
```
3. Run this command step by step for smooth installation of dependencies
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
```bash
# Meta foundational libraries
pip install fvcore iopath omegaconf cloudpickle black hydra-core tensorboard
# Standard vision/evaluation stack
pip install lvis pycocotools scipy shapely pandas opencv-python tqdm timm submitit einops transformers open_clip_torch torchmetrics mmcv==1.7.1 termcolor yapf==0.32.0
```
The original OV-DQUO authors used a deprecated PyTorch C++ API (value.type().is_cuda()) that has been entirely removed from modern PyTorch. Patch their source code before compiling.
```bash
# 1. Fix CUDA assertions
find models/ops/src -type f -exec sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' {} +

# 2. Fix legacy type dispatch macros
find models/ops/src -type f -exec sed -i 's/value\.type()/value.scalar_type()/g' {} +
```
```bash
# Target RTX 5090 natively
export TORCH_CUDA_ARCH_LIST="12.0"

# Build Custom Deformable Attention
cd models/ops
rm -rf build/ dist/
sh make.sh
cd ../../

# Build Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation --force-reinstall --no-deps
```
4. Proposed Method Dependencies (SAM Priors)

To execute the offline SAM prior generation scripts located in tools/sam_priors/, you must install the following:
```bash
pip install ultralytics openpyxl
```

# Data Preparation & Checkpoints
Ensure your directory structure aligns with standard COCO formats. Pre-trained base weights (OVDQUO_RN50_COCO.pth) must be placed in the /ckpt/ directory prior to evaluation.

And please make sure that you have these in your folders, if not, refer to the original OV-DQUO repo to download the weights:
- ckpt/OVDQUO_RN50_COCO.pth
- ckpt/OVDQUO_RN50x4_COCO.pth
- pretrained/region_prompt_R50.pth
- pretrained/region_prompt_R50x4.pth
- ow_labels/OW_COCO_R1.json
- ow_labels/OW_COCO_R2.json
- ow_labels/OW_COCO_R3.json
- ow_labels/OW_LVIS_R3.json

These for the dataset ov-coco, coco2017 images, and also the annotations of original coco and ov-coco annotations
- data/Anotations/captions_train2017.json
- data/Anotations/captions_val2017.json
- data/Anotations/image_info_test-dev2017.json
- data/Anotations/image_info_test2017.json
- data/Anotations/instances_train2017_base.json
- data/Anotations/instances_train2017.json
- data/Anotations/instances_val2017_basetarget.json
- data/Anotations/instances_val2017.json
- data/Anotations/person_keypoints_train2017.json
- data/Anotations/person_keypoints_val2017.json
- data/Images/train2017
- data/Images/val2017
- data/Images/test2017

# Precomputing SAM Priors (Step 1 — before training)

SAMethingOVerDrive replaces the OLN+FE pseudo-labeling pipeline with offline FastSAM segmentation priors. These must be precomputed once before any training run.

Download the FastSAM weights first (if not already present):
```bash
# FastSAM-x.pt will be auto-downloaded by ultralytics on first use,
# or place it manually in the working directory.
```

Run precomputation for both splits (~118k train + 5k val images):
```bash
bash scripts/precompute_priors.sh
```

This writes per-image `.pt` files to `data/sam_priors/train2017/` and `data/sam_priors/val2017/`. Each file contains detection boxes (cx,cy,w,h normalized), 28×28 binary masks, and confidence scores. Images that already have a `.pt` file are skipped, so the script is safe to re-run.

Optional arguments (passed through to the Python script):
```bash
bash scripts/precompute_priors.sh --model /path/to/FastSAM-x.pt --splits train2017
```

## SAMethingOVerDrive: Proposed Method

SAMethingOVerDrive extends [OV-DQUO (AAAI 2025)](https://github.com/xiaomoguhz/OV-DQUO) by replacing its OLN+FE pseudo-labeling pipeline with offline FastSAM segmentation priors, using FastSAM mask confidence as a direct foreground quality signal for wildcard proposal weighting. Novel category localization is further improved through mask-guided CLIP feature purification (Triple-Filter), morphology-aware and CLIP-retrieved semantic wildcards that replace the generic `"object"` token, and FastSAM-augmented Region Query Initialization (RoQIS) that seeds decoder queries directly from SAM-localized regions each forward pass. A detector–SAM disagreement mining buffer (T+1 buffer) additionally recovers novel objects that the detector misses but FastSAM confidently segments, closing the localization gap on unseen categories in the OV-COCO benchmark.

## Precomputation Pipeline

Two precomputation steps must be run **once in order** before any SAMethingOVerDrive training. Outputs are written per-image as `.pt` files; images that already have a file are skipped, so both scripts are safe to resume after interruption.

**Step 1 — FastSAM priors** (~50 min on a single GPU across COCO train2017 + val2017):
```bash
bash scripts/precompute_priors.sh
```
Writes to `data/sam_priors/train2017/<image_id>.pt` and `data/sam_priors/val2017/<image_id>.pt`. Each file contains detection boxes (cx, cy, w, h normalized to [0,1]), 28×28 binary masks, and per-box confidence scores.

**Step 2 — Soft wildcard embeddings** (~20 min on a single GPU, COCO train2017 only):
```bash
bash scripts/precompute_soft_wildcards.sh
```
Writes to `data/soft_wildcards/train2017/<image_id>.pt`. Each file contains per-box CLIP embedding vectors used as the soft wildcard supervision target.

> Step 2 is only required when `use_soft_wildcards=True` (ablations A4–A6 and the full system). A1–A3 can train from Step 1 output alone.

## Training

> **Recommended order:** run A1 first to verify the core SAM-prior hypothesis on your hardware before proceeding to the full system.

**Single-GPU training:**
```bash
# A1 — baseline ablation (SAM priors only)
bash scripts/OV-COCO/train_SAMOVD_A1.sh <output_dir> cuda:0

# Full system (all components enabled)
bash scripts/OV-COCO/train_SAMOVD_RN50.sh <output_dir> cuda:0
```

**Distributed training (8 GPUs):**
```bash
# A1 — baseline ablation
bash scripts/OV-COCO/distrain_SAMOVD_A1.sh <output_dir>

# Full system
bash scripts/OV-COCO/distrain_SAMOVD_RN50.sh <output_dir>
```

Individual ablation scripts `train_SAMOVD_A2.sh` through `train_SAMOVD_A6.sh` (and their `distrain_` counterparts) follow the same `<output_dir> [device]` interface and correspond to the cumulative configs described in the Ablation Study section below.

## Evaluation

Standard evaluation using a trained checkpoint:
```bash
bash scripts/OV-COCO/eval_SAMOVD.sh \
  config/OV_COCO/SAMOVD_RN50_full.py \
  <path/to/checkpoint.pth> \
  cuda:0
```

To enable SAM-calibrated scoring at eval time (rescales detection scores by FastSAM confidence without retraining), invoke `main.py` directly:
```bash
python main.py \
  -c config/OV_COCO/SAMOVD_RN50_full.py \
  --output_dir ./outputs/eval_calibrated \
  --amp \
  --device cuda:0 \
  --resume <path/to/checkpoint.pth> \
  --eval \
  --options sam_calibrate=True
```

`sam_calibrate` defaults to `False` in all configs and can be toggled at eval time without retraining.

## Ablation Study

Results on OV-COCO (RN50 backbone). AP<sup>Novel</sup><sub>50</sub> measures detection AP at IoU=0.50 on novel (unseen) categories.

| Config | Added Component | AP<sup>Novel</sup><sub>50</sub> |
|--------|-----------------|--------------------------------|
| OV-DQUO baseline | — (reproduced) | 39.3 |
| A1 `SAMOVD_RN50_ablation_A1_fastsam_weight.py` | + SAM priors as wildcard proposals | TBD |
| A2 `SAMOVD_RN50_ablation_A2_triple_filter.py` | + Triple-Filter quality gate | TBD |
| A3 `SAMOVD_RN50_ablation_A3_morph_wildcard.py` | + Morphological wildcard labels | TBD |
| A4 `SAMOVD_RN50_ablation_A4_soft_wildcard.py` | + Soft CLIP-similarity wildcard loss | TBD |
| A5 `SAMOVD_RN50_ablation_A5_roqis.py` | + SAM-guided query initialization (RoQIS) | TBD |
| A6 `SAMOVD_RN50_ablation_A6_disagreement.py` | + Detector–SAM disagreement mining | TBD |
| Full `SAMOVD_RN50_full.py` | All components | TBD |

> Results will be updated as experiments complete.

## Method Components

| Component | Addresses | Config Flag | Status |
|-----------|-----------|-------------|--------|
| FastSAM Priors | OV-DQUO dependence on OLN+FE pseudo-labels for novel category localization | `use_sam_priors` | Implemented |
| Triple-Filter | Noisy SAM proposals from invalid box size, aspect ratio, or overlap with GT | `use_triple_filter` | Implemented |
| Morph Wildcards | Generic `"object"` wildcard losing category-discriminative CLIP signal | `use_morph_wildcards` | Implemented |
| Soft Wildcards | Hard binary wildcard target ignoring graded CLIP similarity | `use_soft_wildcards` | Implemented |
| SAM RoQIS | Random query initialization missing SAM-localized foreground regions | `use_sam_roqis` | Implemented |
| Disagreement Mining | Novel objects confidently detected by SAM but missed by the detector | `use_disagreement_mining` | Implemented |

