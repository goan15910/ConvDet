# Author: 

"""Model configuration for ILSVRC2013"""

import numpy as np

from config import base_model_config

def imagenet_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('ILSVRC2013')

  mc.DEBUG_MODE            = False
 
  # Data Augmentation
  mc.DATA_AUGMENTATION     = True

  # Network Architecture
  mc.BN = True
  mc.LOAD_BN = True

  # Remove BGR mean
  mc.SUB_BGR_MEANS = True

  mc.IMAGE_WIDTH           = 224
  mc.IMAGE_HEIGHT          = 224
  mc.BATCH_SIZE            = 32

  mc.WEIGHT_DECAY          = 0.0001
  mc.LEARNING_RATE         = 1e-3
  mc.LR_POLICY             = 'step'
  mc.LR_STEP_BOUNDRY       = [10000, 15000]
  mc.LR_STEP_VALUE         = [1e-3, 1e-4, 1e-5]
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9

  return mc
