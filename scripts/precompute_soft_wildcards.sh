#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SAMOVD

python tools/sam_priors/build_vocab_embeddings.py "$@"
python tools/sam_priors/precompute_soft_wildcards.py "$@"
