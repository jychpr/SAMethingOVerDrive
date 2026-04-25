#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SAMOVD

python tools/sam_priors/precompute_sam_coco.py "$@"
