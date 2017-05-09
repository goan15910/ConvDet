# Author: 

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def pascal_voc_yolo_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('PASCAL_VOC')

  mc.DEBUG_MODE            = True
 
  mc.SUB_BGR_MEANS = False

  # Data Augmentation
  mc.LOSS_TYPE = 'YOLO'
  mc.DATA_AUG_TYPE = 'YOLO'

  # Network Architecture
  mc.BN = True
  mc.LOAD_BN = True

  mc.IMAGE_WIDTH           = 416
  mc.IMAGE_HEIGHT          = 416
  mc.BATCH_SIZE            = 32

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 1e-3
  mc.LR_POLICY             = 'step'
  mc.LR_STEP_BOUNDRY       = [10000, 15000]
  mc.LR_STEP_VALUE         = [1e-3, 1e-4, 1e-5]
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9

  mc.LOSS_COEF_BBOX        = 1.0
  mc.LOSS_COEF_CONF_POS    = 5.0
  mc.LOSS_COEF_CONF_NEG    = 1.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.4

  mc.DATA_AUGMENTATION     = True

  mc.NET_OUT_SHAPE         = (13, 13, 5) # (H, W, B)
  mc.ANCHOR_BOX            = set_anchors(mc, mc.NET_OUT_SHAPE[-1])
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 5

  return mc


def set_anchors(mc, B):
  anchors = np.reshape(
      np.array(
          [[ 1.08, 1.19 ], [ 3.42, 4.41 ], [ 6.63, 11.38 ],
           [ 9.42, 5.11 ], [ 16.62, 10.52 ]]
      ),
      (1, 1, B, 2)
  )
  return anchors
