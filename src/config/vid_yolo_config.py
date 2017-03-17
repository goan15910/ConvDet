# Author: 

"""Model configuration for VID dataset"""

import numpy as np

from config import base_model_config

def vid_yolo_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('VID')

  mc.DEBUG_MODE            = False
  
  # Data Augmentation
  mc.LOSS_TYPE = 'YOLO'
  mc.DATA_AUG_TYPE = 'YOLO'

  # Network Architecture
  mc.BN = True

  mc.IMAGE_WIDTH           = 416
  mc.IMAGE_HEIGHT          = 416
  mc.BATCH_SIZE            = 64

  mc.WEIGHT_DECAY          = 1e-4
  mc.LEARNING_RATE         = 1e-3
  mc.DECAY_STEPS           = 2e4
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

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
  mc.ANCHOR_PER_GRID       = 9

  return mc

#TODO(jeff): customize anchors for pascal voc
def set_anchors(mc):
  H, W, B = 13, 13, 9
  anchor_shapes = np.reshape(
      [np.array(
          [[  100.,  70.], [ 200., 140.], [ 400.,  280.],
           [ 92.,  92.], [  185.,  185.], [ 370., 370.],
           [ 64., 96.], [ 128., 192.], [  256.,  384.]])] * H * W,
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
