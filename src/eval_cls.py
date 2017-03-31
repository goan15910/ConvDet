
"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti, vid, imagenet
from utils.util import Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'ILSVRC2013',
                           """ILSVRC2013 only""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val',
                           """Only used for VOC data."""
                           """train, val, or test""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp3/jeff/ConvDet/experiments/eval_val',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('pkl_path', '/tmp3/jeff/ConvDet/data/darknet/darknet19_weights_bgr.pkl',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_string('net', 'darknet19',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def evaluate():
  """Evaluate."""
  assert FLAGS.dataset == 'ILSVRC2013', \
      'Only ILSVRC2013'

  with tf.Graph().as_default() as g:

    mc = imagenet_config()
    mc.PRETRAINED_MODEL_PATH = FLAGS.pkl_path
    imdb = imagenet(FLAGS.image_set, FLAGS.data_path, mc)

    assert FLAGS.net == 'darknet19', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    model = DARKNET19(mc, FLAGS.gpu)
    
    init = tf.initialize_all_variables()
    
    # TODO(jeff): add cls inference & evaluation
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      # run init
      sess.run(init)

      # testing
      num_images = len(imdb.image_idx)
      num_batches = np.ceil(float(num_images) / mc.BATCH_SIZE).astype(np.int64)

      _t = {'im_cls': Timer(), 'im_read': Timer()}

      all_labels, all_preds = [], []
      for i in xrange(num_batches):
        print '{} / {}'.format(i+1, num_batches)
        _t['im_read'].tic()
        images, labels, scales = imdb.read_cls_batch(shuffle=False)
        _t['im_read'].toc()

        _t['im_cls'].tic()
        cls_idx = sess.run(
            [model.pred_class],
            feed_dict={model.image_input:images, \
                       model.is_training: False})
        _t['im_cls'].toc()
        all_labels.extend(labels)
        all_preds.extend(cls_idx[0].tolist())

      # evaluate
      acc = 0.
      for i in xrange(num_images):
        if all_labels[i] == all_preds[i]+1:
          acc += 1.
      acc = acc * 100. / num_images
      print 'Evaluation:'
      print '  Timing:'
      print '    im_read: {:.3f}s im_cls: {:.3f}'.format( \
        _t['im_read'].average_time / mc.BATCH_SIZE, \
        _t['im_cls'].average_time / mc.BATCH_SIZE)
      print '  Accuracy: {:.2f}%'.format(acc)
      

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
