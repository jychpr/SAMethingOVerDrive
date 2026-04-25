#!/bin/bash

config=$1
checkpoint=$2
device=${3:-cuda:0}

echo "=== Evaluating config:     $config ==="
echo "=== Checkpoint:            $checkpoint ==="

python main.py \
    -c $config \
    --output_dir ./outputs/eval_$(basename $config .py) \
    --amp \
    --device $device \
    --resume $checkpoint \
    --eval
