#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=GPU ./scripts/eval_train.sh GPU DATASET

export PYTHONUNBUFFERED="True"

GPU=$1
DATASET=$2

#NET=vgg16
NET=vgg16_v2
#NET=vgg16_v3

case $DATASET in
  PASCAL_VOC)
    echo "Use PASCAL_VOC dataset"
    DATA_PATH=/tmp3/jeff/VOCdevkit2007
    CKPT_PATH=./experiments/vgg16/train/PASCAL_VOC
    IMAGE_SET=train
    ;;
  VID)
    echo "Use VID dataset"
    DATA_PATH=/tmp3/jeff/vid/ILSVRC2015
    IMAGE_SET=train
    ;;
  KITTI)
    echo "Use KITTI dataset"
    DATA_PATH=data/KITTI
    CKPT_PATH=./experiments/vgg16/train/KITTI
    IMAGE_SET=train
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

EVAL_DIR=experiments/vgg16/eval_train/${DATASET}
if [ ! -d "$EVAL_DIR" ]; then
    mkdir "$EVAL_DIR"
fi

LOG_DIR=experiments/logs
LOG="$LOG_DIR/${NET}_${DATASET}_eval_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
 python ./src/eval.py \
   --dataset=${DATASET} \
   --data_path=${DATA_PATH} \
   --image_set=${IMAGE_SET} \
   --eval_dir=${EVAL_DIR} \
   --checkpoint_path=${CKPT_PATH} \
   --net=${NET} \
   --gpu=${GPU}
