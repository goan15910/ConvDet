
"""Classification Demo. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob 

import numpy as np
import tensorflow as tf
import joblib

from config import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'gpu', '0',"""GPU id.""")
tf.app.flags.DEFINE_string(
    'net', 'darknet19',"""Model type.""")
tf.app.flags.DEFINE_string(
    'pkl_path', '',"""Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'meta_path', '',"""Path to the dataset meta file.""")
tf.app.flags.DEFINE_string(
    'input_path', '',
    """Input image(s). Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', '',"""Path to output directory.""")


def _draw_cls_label(im, label, cdict=None):
  h, w, _ = im.shape
  if cdict and (l in cdict):
    c = cdict[l]
  else:
    c = (0, 0, 255)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(im, label, (0, h-5), font, 0.5, c, 2)


def image_demo():
  """Classify image."""

  with tf.Graph().as_default():
    # Load model
    mc = imagenet_config()
    mc.PRETRAINED_MODEL_PATH = FLAGS.pkl_path
    mc.BATCH_SIZE = 1

    assert FLAGS.net in ['darknet19', 'vgg16'], \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'darknet19':
      model = DARKNET19(mc, FLAGS.gpu)
    elif FLAGS.net == 'vgg16':
      model = VGG16(mc, FLAGS.gpu)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      # run init
      init = tf.global_variables_initializer()
      sess.run(init)

      for i,f in enumerate(glob.iglob(FLAGS.input_path)):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        input_image = im - mc.BGR_MEANS

        # classifiy
        cls_idx = sess.run([model.pred_class], \
            feed_dict={model.image_input: [input_image], \
                       model.is_training: False})
        pred = (cls_idx[0] + 1).tolist()

        # Draw label
        meta_d = joblib.load(FLAGS.meta_path)
        label = meta_d[pred[0]]
        _draw_cls_label(im, label)

        # Dump 
        out_fname = os.path.join(FLAGS.out_dir, '{}.jpg'.format(i+1))
        cv2.imwrite(out_fname, im)


def main(argv=None):
  image_demo()

if __name__ == '__main__':
    tf.app.run()
