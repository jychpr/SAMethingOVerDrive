# SAMethingOVerDrive
SAM Priors Localization for Open Vocabulary Detection

SAMethingOVerDrive extends [OV-DQUO (AAAI 2025)](https://github.com/xiaomoguhz/OV-DQUO) by replacing its OLN+FE pseudo-labeling pipeline with offline FastSAM segmentation priors, using FastSAM confidence as a foreground quality signal for wildcard proposal weighting. Novel category localization is further improved through mask-guided CLIP feature purification (Triple-Filter), morphology-aware and CLIP-retrieved semantic wildcards that replace the generic `"object"` token, and FastSAM-augmented Region Query Initialization (RoQIS) that seeds decoder queries directly from SAM-localized regions each forward pass. A detector–SAM disagreement mining buffer (T+1 buffer) additionally recovers novel objects that the detector misses but FastSAM confidently segments, closing the localization gap on unseen categories in the OV-COCO benchmark. All six components are individually toggleable via config flags and evaluated in a cumulative ablation over the OV-DQUO baseline.

## Table of Contents
- [Initial Setup](#initial-setup)
  - [Environment and Dependencies Setup](#environment-and-dependencies-setup)
  - [Data Preparation & Checkpoints](#data-preparation--checkpoints)
  - [Downloading Required Files (CLI / SSH)](#downloading-required-files-cli--ssh)
- [How to Run](#how-to-run)
  - [Precomputation Pipeline](#precomputation-pipeline)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
  - [Ablation Study](#ablation-study)
  - [Method Components](#method-components)

# Initial Setup
## Environment and Dependencies Setup
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

## Data Preparation & Checkpoints
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

If the above hasn't been done, please check the following section

## Downloading Required Files (CLI / SSH)

This section is for headless or SSH environments. Install gdown first:

```bash
pip install gdown
```

Then run the following commands from the repo root to download and place all required files.

**OV-DQUO Checkpoints**
```bash
mkdir -p ckpt
gdown "https://drive.google.com/uc?id=17Nlo0V4jrJz0bNvivfFXcOcaYZq-Up3x" -O ckpt/OVDQUO_RN50_COCO.pth
gdown "https://drive.google.com/uc?id=1bDxIj1spUmqrMRNHGzK5TZd9uhL9T1KG" -O ckpt/OVDQUO_RN50x4_COCO.pth
```

**Pretrained Region-Prompt Weights**
```bash
mkdir -p pretrained
gdown --folder "https://drive.google.com/drive/folders/17mi8O1YW6dl8TRkwectHRoC8xbK5sLMw" -O pretrained/
```

**OW Label JSONs**
```bash
mkdir -p ow_labels
gdown --folder "https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4" -O ow_labels/
```

**OV-COCO Annotations**
```bash
mkdir -p data/Anotations
gdown --folder "https://drive.google.com/drive/folders/1Jgkpoz_ILJRI4xRJydi7dQfFjwtAFbef" -O data/Anotations/
```

> ⚠ **Disk space:** COCO 2017 requires approximately 26 GB of free disk space (19 GB train images, 1 GB val/test images, ~1 GB annotations). Verify available space with `df -h .` before starting.

**COCO 2017 Original Dataset (Images + Annotations)**
```bash
mkdir -p data/Images data/Anotations

# Images
wget -P data/Images/ http://images.cocodataset.org/zips/train2017.zip
wget -P data/Images/ http://images.cocodataset.org/zips/val2017.zip
wget -P data/Images/ http://images.cocodataset.org/zips/test2017.zip

# Annotations
wget -P data/Anotations/ http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -P data/Anotations/ http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -P data/Anotations/ http://images.cocodataset.org/annotations/image_info_test-dev2017.zip

# Unzip
unzip data/Images/train2017.zip  -d data/Images/
unzip data/Images/val2017.zip    -d data/Images/
unzip data/Images/test2017.zip   -d data/Images/
unzip data/Anotations/annotations_trainval2017.zip -d data/Anotations/
unzip data/Anotations/image_info_test2017.zip      -d data/Anotations/
unzip data/Anotations/image_info_test-dev2017.zip  -d data/Anotations/

# Optional: remove zips to save space
rm data/Images/*.zip data/Anotations/*.zip
```

> **Note on gdown folder downloads:** if you hit a Google Drive quota or rate-limit error, add `--remaining-ok` to skip already-downloaded files. For very large folders, consider using `rclone` with a Google service account for more reliable transfers.


# How to Run

## Precomputation Pipeline

SAMethingOVerDrive replaces the OLN+FE pseudo-labeling pipeline with offline FastSAM segmentation priors. These must be precomputed once before any training run.

Download the FastSAM weights first (if not already present). On SSH or headless servers, auto-download may fail silently — use the explicit wget command:

```bash
# Option 1 — explicit download (recommended for SSH/headless environments)
wget -O FastSAM-x.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt

# Option 2 — ultralytics will auto-download on first script run if the file is absent
# (requires outbound internet access from the compute node)
```

Two precomputation steps must be run **once in order** before any SAMethingOVerDrive training. Outputs are written per-image as `.pt` files; images that already have a file are skipped, so both scripts are safe to resume after interruption.

**Step 1 — FastSAM priors** (~50 min on a single GPU across COCO train2017 + val2017):
```bash
bash scripts/precompute_priors.sh
```
Writes to `data/sam_priors/train2017/<image_id>.pt` and `data/sam_priors/val2017/<image_id>.pt`. Each file contains detection boxes (cx, cy, w, h normalized to [0,1]), 28×28 binary masks, and per-box confidence scores.

> ✓ **Sanity check:** `data/sam_priors/train2017/` should contain ~118,287 `.pt` files and `data/sam_priors/val2017/` should contain ~5,000 `.pt` files. Verify with `ls data/sam_priors/train2017/ | wc -l`.

**Step 2 — Soft wildcard embeddings** (~20 min on a single GPU, COCO train2017 only):
```bash
bash scripts/precompute_soft_wildcards.sh
```
Writes to `data/soft_wildcards/train2017/<image_id>.pt`. Each file contains per-box CLIP embedding vectors used as the soft wildcard supervision target.

> ✓ **Sanity check:** `data/soft_wildcards/train2017/` should contain ~118,287 `.pt` files. Verify with `ls data/soft_wildcards/train2017/ | wc -l`.

> Step 2 is only required when `use_soft_wildcards=True` (ablations A4–A6 and the full system). A1–A3 can train from Step 1 output alone.

## Training

> **Recommended order:** run A1 first to verify the core SAM-prior hypothesis on your hardware before proceeding to the full system.
>
> **GPU requirements:** single-GPU training requires a minimum of 24 GB VRAM (tested on RTX 5090). For GPUs with less VRAM, use the distributed training commands and reduce `--batch_size` accordingly.

> **`<output_dir>`** is the path where checkpoints, logs, and evaluation results will be saved. It is created automatically if it does not exist. Use a descriptive name, e.g. `outputs/ablation_A1` or `outputs/full_system_run1`.

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

Sample command for training (please adjust it accordingly)
```bash
bash scripts/OV-COCO/train_SAMOVD_A1.sh outputs/ablation_A1 cuda:0
```

## Evaluation

**Standard evaluation** using the shell script:

```bash
bash scripts/OV-COCO/eval_SAMOVD.sh \
  config/OV_COCO/SAMOVD_RN50_full.py \
  <path/to/checkpoint.pth> \
  cuda:0
```

**Evaluation with SAM calibration** — rescales detection scores by FastSAM confidence at inference time, no retraining required. Pass the option directly via `main.py`:

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

> `sam_calibrate` defaults to `False` in all configs and can be toggled at eval time without retraining.

# Results

## Ablation Study

Results on OV-COCO (RN50 backbone). AP<sup>Novel</sup><sub>50</sub> measures detection AP at IoU=0.50 on novel (unseen) categories.

| Config | Description | AP<sup>Novel</sup><sub>50</sub> |
|--------|-------------|--------------------------------|
| OV-DQUO Baseline | Original OV-DQUO (RN50, COCO) | 39.3 |
| A1 | SAM priors as wildcard proposals (`use_sam_priors`) | TBD |
| A2 | + Triple-Filter quality gate (`use_triple_filter`) | TBD |
| A3 | + Morphological wildcard labels (`use_morph_wildcards`) | TBD |
| A4 | + Soft CLIP-similarity wildcard loss (`use_soft_wildcards`) | TBD |
| A5 | + SAM-guided query initialization / RoQIS (`use_sam_roqis`) | TBD |
| A6 | + Detector–SAM disagreement mining (`use_disagreement_mining`) | TBD |
| Full System | All components enabled | TBD |

> Results will be updated as experiments complete.

## Method Components

| Component | Addresses | Config Flag | Status |
|-----------|-----------|-------------|--------|
| FastSAM Priors | Replaces OLN+FE pseudo-labeling | `use_sam_priors` | Implemented |
| Triple-Filter | Mask-guided CLIP feature purification | `use_triple_filter` | Implemented |
| Morph Wildcards | Morphology-aware semantic wildcards | `use_morph_wildcards` | Implemented |
| Soft Wildcards | CLIP-retrieved semantic wildcards | `use_soft_wildcards` | Implemented |
| SAM RoQIs | FastSAM-augmented RoI quality selection | `use_sam_roqis` | Implemented |
| Disagreement Mining | Detector-SAM disagreement as unknown signal | `use_disagreement_mining` | Implemented |

