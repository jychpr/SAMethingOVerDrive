#!/bin/bash

output_dir=$1
device=${2:-cuda:0}
python main.py \
	--output_dir $output_dir -c config/OV_COCO/SAMOVD_RN50_ablation_A1_fastsam_weight.py \
	--amp \
	--device $device \
	--eval_start_epoch 5 \
	--eval_every_epoch 5 \
