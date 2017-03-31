
"""VGG16 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton


class VGG16(ModelSkeleton):
  def __init__(self, mc, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self.BN = mc.BN
      self._add_forward_graph()
      self._add_cls_interpretation_graph()

  def _add_cls_interpretation_graph(self):
    """Inference logits"""
    self.pred_class_probs = tf.nn.softmax(
        tf.reshape(
            self.preds,
            [-1, self.mc.CLASSES]
        ),
        name='pred_class_probs'
    )

    self.pred_class = tf.argmax(
        self.pred_class_probs,
        axis=1,
        name='class_idx'
    )

  def _add_forward_graph(self):
    """Build the VGG16 model."""

    if self.mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(self.mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(self.mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(self.mc.PRETRAINED_MODEL_PATH)

    with tf.variable_scope('conv1') as scope:
      conv1_1 = self._conv_layer(
          'conv1_1', self.image_input, filters=64, size=3, stride=1, freeze=True)
      conv1_2 = self._conv_layer(
          'conv1_2', conv1_1, filters=64, size=3, stride=1, freeze=True)
      pool1 = self._pooling_layer(
          'pool1', conv1_2, size=2, stride=2)

    with tf.variable_scope('conv2') as scope:
      conv2_1 = self._conv_layer(
          'conv2_1', pool1, filters=128, size=3, stride=1, freeze=True)
      conv2_2 = self._conv_layer(
          'conv2_2', conv2_1, filters=128, size=3, stride=1, freeze=True)
      pool2 = self._pooling_layer(
          'pool2', conv2_2, size=2, stride=2)

    with tf.variable_scope('conv3') as scope:
      conv3_1 = self._conv_layer(
          'conv3_1', pool2, filters=256, size=3, stride=1)
      conv3_2 = self._conv_layer(
          'conv3_2', conv3_1, filters=256, size=3, stride=1)
      conv3_3 = self._conv_layer(
          'conv3_3', conv3_2, filters=256, size=3, stride=1)
      pool3 = self._pooling_layer(
          'pool3', conv3_3, size=2, stride=2)

    with tf.variable_scope('conv4') as scope:
      conv4_1 = self._conv_layer(
          'conv4_1', pool3, filters=512, size=3, stride=1)
      conv4_2 = self._conv_layer(
          'conv4_2', conv4_1, filters=512, size=3, stride=1)
      conv4_3 = self._conv_layer(
          'conv4_3', conv4_2, filters=512, size=3, stride=1)
      pool4 = self._pooling_layer(
          'pool4', conv4_3, size=2, stride=2)

    with tf.variable_scope('conv5') as scope:
      conv5_1 = self._conv_layer(
          'conv5_1', pool4, filters=512, size=3, stride=1)
      conv5_2 = self._conv_layer(
          'conv5_2', conv5_1, filters=512, size=3, stride=1)
      conv5_3 = self._conv_layer(
          'conv5_3', conv5_2, filters=512, size=3, stride=1)
      pool5 = self._pooling_layer(
          'pool5', conv5_3, size=2, stride=2)

    with tf.variable_scope('fc') as scope:
      fc6 = self._fc_layer(
          'fc6', pool5, 4096, flatten=True)
      fc7 = self._fc_layer(
          'fc7', fc6, 4096)
      self.preds = self._fc_layer(
          'fc8', fc7, 1000, relu=False)
