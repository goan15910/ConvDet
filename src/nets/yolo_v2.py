
"""YOLO-v2 model."""

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


class YOLO_V2(ModelSkeleton):
  def __init__(self, mc, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self.BN = mc.BN
      self._add_forward_graph()
      self._add_yolo_interpret_graph()
      #self._add_yolo_loss_graph()
      #self._add_train_graph()
      #self._add_viz_graph()

  def _add_forward_graph(self):
    """Build the VGG-16 model."""

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

    with tf.variable_scope('detector') as scope:
      conv19 = self._conv_layer(
          'conv19', conv18, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      conv20 = self._conv_layer(
          'conv20', conv19, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      reorg20 = self._reorg_layer('reorg20', conv13, stride=2)
      concat20 = self._concat_layer('concat20', conv20, reorg20)
      conv21 = self._conv_layer(
          'conv21', concat20, filters=1024, size=3, stride=1, bn=self.BN, act='lrelu')
      num_output = self.mc.ANCHOR_PER_GRID * (self.mc.CLASSES + 1 + 4)
      self.preds = self._conv_layer(
          'conv22', conv21, filters=num_output, size=1, stride=1,
          padding='SAME', xavier=False, act=None, stddev=0.0001)
    self.conv13 = conv13
    self.conv20 = conv20
    self.reorg20 = reorg20
    self.concat20 = concat20
