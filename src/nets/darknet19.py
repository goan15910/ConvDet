
"""Darknet19 model."""

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


class DARKNET19(ModelSkeleton):
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
            [-1, mc.CLASSES]
        ),
        name='pred_class_probs'
    )

    self.pred_class = tf.argmax(
        self.pred_class_probs,
        axis=0,
        name='class_idx'
    )

  def _add_forward_graph(self):
    """Build the Darknet19 model."""

    if self.mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(self.mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(self.mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(self.mc.PRETRAINED_MODEL_PATH)

    with tf.variable_scope('darknet19') as scope:
      conv1 = self._conv_layer(
          'conv1', self.image_input, filters=32, size=3, stride=1, bn=self.BN, act='lrelu', freeze=True)
      pool1 = self._pooling_layer(
          'pool1', conv1, size=2, stride=2)
      conv2 = self._conv_layer(
          'conv2', pool1, filters=64, size=3, stride=1, bn=self.BN, act='lrelu', freeze=True)
      pool2 = self._pooling_layer(
          'pool2', conv2, size=2, stride=2)
      conv3 = self._conv_layer(
          'conv3', pool2, filters=128, size=3, stride=1, bn=self.BN, act='lrelu')
      conv4 = self._conv_layer(
          'conv4', conv3, filters=64, size=1, stride=1, bn=self.BN, act='lrelu')
      conv5 = self._conv_layer(
          'conv5', conv4, filters=128, size=3, stride=1, bn=self.BN, act='lrelu')
      pool3 = self._pooling_layer(
          'pool3', conv5, size=2, stride=2)
      conv6 = self._conv_layer(
          'conv6', pool3, filters=256, size=3, stride=1, bn=self.BN, act='lrelu')
      conv7 = self._conv_layer(
          'conv7', conv6, filters=128, size=1, stride=1, bn=self.BN, act='lrelu')
      conv8 = self._conv_layer(
          'conv8', conv7, filters=256, size=3, stride=1, bn=self.BN, act='lrelu')
      pool4 = self._pooling_layer(
          'pool4', conv8, size=2, stride=2)
      conv9 = self._conv_layer(
          'conv9', pool4, filters=512, size=3, stride=1, bn=self.BN, act='lrelu')
      conv10 = self._conv_layer(
          'conv10', conv9, filters=256, size=1, stride=1, bn=self.BN, act='lrelu')
      conv11 = self._conv_layer(
          'conv11', conv10, filters=512, size=3, stride=1, bn=self.BN, act='lrelu')
      conv12 = self._conv_layer(
          'conv12', conv11, filters=256, size=1, stride=1, bn=self.BN, act='lrelu')
      conv13 = self._conv_layer(
          'conv13', conv12, filters=512, size=3, stride=1, bn=self.BN, act='lrelu')
      pool5 = self._pooling_layer(
          'pool5', conv13, size=2, stride=2)
      conv14 = self._conv_layer(
          'conv14', pool5, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      conv15 = self._conv_layer(
          'conv15', conv14, filters=512, size=1, stride=1, bn=self.BN, act='lrelu')
      conv16 = self._conv_layer(
          'conv16', conv15, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      conv17 = self._conv_layer(
          'conv17', conv16, filters=512, size=1, stride=1, bn=self.BN, act='lrelu')
      conv18 = self._conv_layer(
          'conv18', conv17, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      conv19 = self._conv_layer(
          'conv19', conv18, filters=self.mc.CLASSES, size=1, stride=1,
          padding='SAME', xavier=False, act=None, stddev=0.0001)
      self.preds = self._pooling_layer(
          'global_pool', conv19, size=None, stride=None, ptype='global_avg')
