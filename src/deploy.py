
"""Deploy"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time
import math

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti, vid
from utils.util import bbox_transform, Timer
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'PASCAL_VOC',
                           """PASCAL_VOC / KITTI""")
tf.app.flags.DEFINE_string('data_path', '/tmp3/jeff/VOCdevkit2007', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp3/jeff/ConvDet/experiments/yolo_v2/deploy',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('pretrained_model_path', '/tmp3/jeff/ConvDet/data/yolo/yolo_weights.pkl',
                            """Path to pretrained weight path.""")
tf.app.flags.DEFINE_string('net', 'yolo_v2',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def eval_once(saver, summary_writer, imdb, model, mc):

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    #global_step = '0'
    global_step = None

    n_imgs = len(imdb.image_idx)
    n_iters = int(n_imgs / mc.BATCH_SIZE) + 1

    all_boxes = [[[] for _ in xrange(n_imgs)]
                 for _ in xrange(imdb.num_classes)]

    _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}

    num_detection = 0.0
    for i in xrange(n_iters):
      _t['im_read'].tic()
      images, scales = imdb.read_image_batch(shuffle=False)
      _t['im_read'].toc()

      _t['im_detect'].tic()
      # TODO(jeff): remove output other than det_boxes, det_probs, det_class
      det_boxes, det_probs, det_class, probs, confs, \
        conv13, reorg20, concat20 = sess.run(
          [
            model.det_boxes, model.det_probs, model.det_class, 
            model.probs, model.pred_conf,
            model.conv13, model.reorg20, model.concat20
          ],
          feed_dict={model.image_input:images, \
                       model.is_training: False, model.keep_prob: 1.0}
        )
      _t['im_detect'].toc()

      _t['misc'].tic()
      for j in range(len(det_boxes)): # batch
        # rescale
        det_boxes[j, :, 0::2] /= scales[j][0]
        det_boxes[j, :, 1::2] /= scales[j][1]

        det_bbox, score, det_class = model.filter_yolo_predict(
            det_boxes[j], det_probs[j], det_class[j])

        num_detection += len(det_bbox)
        for c, b, s in zip(det_class, det_bbox, score):
          all_boxes[c][i].append(bbox_transform(b) + [s])
      _t['misc'].toc()

      print ('im_detect: {:d}/{:d} im_read: {:.3f}s '
             'detect: {:.3f}s misc: {:.3f}s'.format(
                i+1, n_imgs, _t['im_read'].average_time,
                _t['im_detect'].average_time, _t['misc'].average_time))

    print ('Evaluating detections...')
    aps, ap_names = imdb.evaluate_detections(
        FLAGS.eval_dir, global_step, all_boxes)

    print ('Evaluation summary:')
    print ('  Average number of detections per image: {}:'.format(
      num_detection/n_imgs))
    print ('  Timing:')
    print ('    im_read: {:.3f}s detect: {:.3f}s misc: {:.3f}s'.format(
      _t['im_read'].average_time, _t['im_detect'].average_time,
      _t['misc'].average_time))
    print ('  Average precisions:')

    eval_summary_ops = []
    for cls, ap in zip(ap_names, aps):
      eval_summary_ops.append(
          tf.summary.scalar('APs/'+cls, ap)
      )
      print ('    {}: {:.3f}'.format(cls, ap))
    print ('    Mean average precision: {:.3f}'.format(np.mean(aps)))
    eval_summary_ops.append(
        tf.summary.scalar('APs/mAP', np.mean(aps))
    )
    eval_summary_ops.append(
        tf.summary.scalar('timing/image_detect', _t['im_detect'].average_time)
    )
    eval_summary_ops.append(
        tf.summary.scalar('timing/image_read', _t['im_read'].average_time)
    )
    eval_summary_ops.append(
        tf.summary.scalar('timing/post_process', _t['misc'].average_time)
    )
    eval_summary_ops.append(
        tf.summary.scalar('num_detections_per_image', num_detection/n_imgs)
    )

    print ('Analyzing detections...')
    stats, ims = imdb.do_detection_analysis_in_eval(
        FLAGS.eval_dir, global_step)
    for k, v in stats.iteritems():
      eval_summary_ops.append(
          tf.summary.scalar(
            'Detection Analysis/'+k, v)
      )

    eval_summary_str = sess.run(eval_summary_ops)
    for sum_str in eval_summary_str:
      summary_writer.add_summary(sum_str, global_step)

def evaluate():
  """Evaluate."""
  assert FLAGS.dataset in ['PASCAL_VOC', 'VID'], \
      'Either PASCAL_VOC / VID'
  if FLAGS.dataset == 'PASCAL_VOC':
    mc = pascal_voc_yolo_config()
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
    mc.LOAD_PRETRAINED_MODEL = True
    imdb = pascal_voc(FLAGS.image_set, FLAGS.year, FLAGS.data_path, mc)
  elif FLAGS.dataset == 'VID':
    mc = vid_yolo_config()
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
    mc.LOAD_PRETRAINED_MODEL = True
    imdb = vid(FLAGS.image_set, FLAGS.data_path, mc)

  with tf.Graph().as_default() as g:

    assert FLAGS.net == 'yolo_v2', \
        'Support only yolo_v2'
    model = YOLO_V2(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    
    # Evaluate only once for deployment
    eval_once(saver, summary_writer, imdb, model, mc)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
