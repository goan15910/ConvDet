# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""VGG16-ConvDet-v2 model."""

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


class VGG16ConvDetV2(ModelSkeleton):
  def __init__(self, mc, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      assert mc.LOSS_TYPE in ['SQT', 'YOLO'], \
          'Loss type {0} not defined'.format(mc.LOSS_TYPE)
      if mc.LOSS_TYPE == 'SQT':
        self._add_sqt_loss_graph()
      elif mc.LOSS_TYPE == 'YOLO':
        self._add_yolo_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """Build the VGG-16 model."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

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

    with tf.variable_scope('conv6') as scope:
      conv6_1 = self._conv_layer(
          'conv6_1', pool5, filters=1024, size=3, stride=1)
      conv6_2 = self._conv_layer(
          'conv6_2', conv6_1, filters=1024, size=3, stride=1)

    dropout6 = tf.nn.dropout(conv6_2, self.keep_prob, name='drop6')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv7', dropout6, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)
