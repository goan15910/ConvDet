#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=GPU ./scripts/train.sh GPU WEIGHT DATASET

export PYTHONUNBUFFERED="True"

GPU=$1
WEIGHT=$2
DATASET=$3

#NET=vgg16
#NET=vgg16_v2
NET=vgg16_v3

case $DATASET in
  PASCAL_VOC)
    echo "Use PASCAL_VOC dataset"
    DATA_PATH=/tmp3/jeff/VOCdevkit2007
    TRAIN_SET=trainval
    VAL_SET=test
    ;;
  VID)
    echo "Use VID dataset"
    DATA_PATH=/tmp3/jeff/vid/ILSVRC2015
    TRAIN_SET=train
    VAL_SET=val
    ;;
  KITTI)
    echo "Use KITTI dataset"
    DATA_PATH=data/KITTI
    TRAIN_SET=train
    VAL_SET=val
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

DEBUG=false
TRAIN_DIR=experiments/vgg16/train/${DATASET}
if [ ! -d "$TRAIN_DIR" ]; then
    mkdir "$TRAIN_DIR"
fi

LOG_DIR=experiments/logs
if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

LOG="$LOG_DIR/${NET}_${DATASET}_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# =========================================================================== #
# command for VGG16:
# =========================================================================== #

if [ $DEBUG = true ]; then
  python -m ipdb ./src/train.py \
   --dataset=${DATASET} \
   --pretrained_model_path=${WEIGHT} \
   --data_path=${DATA_PATH} \
   --train_set=${TRAIN_SET} \
   --val_set=${VAL_SET} \
   --train_dir=${TRAIN_DIR} \
   --net=${NET} \
   --summary_step=100 \
   --checkpoint_step=500 \
   --gpu=${GPU}
else
  python ./src/train.py \
   --dataset=${DATASET} \
   --pretrained_model_path=${WEIGHT} \
   --data_path=${DATA_PATH} \
   --train_set=${TRAIN_SET} \
   --val_set=${VAL_SET} \
   --train_dir=${TRAIN_DIR} \
   --net=${NET} \
   --summary_step=100 \
   --checkpoint_step=500 \
   --gpu=${GPU}
fi
