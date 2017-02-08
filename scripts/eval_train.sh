#!/bin/bash
# Usage: ./scripts/eval_train.sh GPU

GPU=$1

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
 python ./src/eval.py \
   --dataset=KITTI \
   --data_path=./data/KITTI \
   --image_set=train \
   --eval_dir=./logs/vgg16/eval_train \
   --checkpoint_path=./logs/vgg16/train \
   --net=vgg16 \
   --gpu=${GPU}
