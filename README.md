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

