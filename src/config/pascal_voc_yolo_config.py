# Author: 

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def pascal_voc_yolo_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('PASCAL_VOC')

  mc.DEBUG_MODE            = False
 
  #mc.SUB_BGR_MEANS = False

  # Data Augmentation
  mc.LOSS_TYPE = 'YOLO'
  mc.DATA_AUG_TYPE = 'YOLO'

  # Network Architecture
  mc.BN = True

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
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 5

  return mc

def set_anchors(mc):
  H, W, B = 13, 13, 5
  anchor_shapes = np.reshape(
      [np.array(
          [[  240.,  150.], [ 60., 80.], [ 250.,  350.],
           [ 420.,  260.], [  120.,  210.]])] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
