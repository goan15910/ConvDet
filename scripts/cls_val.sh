#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=GPU ./scripts/cls_val.sh GPU DATASET

export PYTHONUNBUFFERED="True"

GPU=$1
NET=$2
DATASET=$3

case $NET in
  darknet19)
    echo "Use Darknet19"
    PKL_PATH=./data/darknet/darknet19_weights_bn_bgr.pkl
    ;;
  vgg16)
    echo "Use VGG16"
    PKL_PATH=./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

case $DATASET in
  ILSVRC2013)
    echo "Use ILSVRC2013 dataset"
    DATA_PATH=/tmp3/jeff/ILSVRC2013
    IMAGE_SET=val
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

EVAL_DIR=experiments/${NET}/cls_val/${DATASET}
if [ ! -d "$EVAL_DIR" ]; then
    mkdir "$EVAL_DIR"
fi

LOG_DIR=experiments/logs
LOG="$LOG_DIR/${NET}_${DATASET}_eval_val.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
 python ./src/eval_cls.py \
   --dataset=${DATASET} \
   --data_path=${DATA_PATH} \
   --image_set=${IMAGE_SET} \
   --eval_dir=${EVAL_DIR} \
   --pkl_path=${PKL_PATH} \
   --net=${NET} \
   --gpu=${GPU}
