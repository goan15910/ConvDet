# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Test imdb functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti, vid
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
import matplotlib.pyplot as plt
import joblib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'PASCAL_VOC',
                           """KITTI / PASCAL_VOC / VID""")
tf.app.flags.DEFINE_string('data_path', '/tmp3/jeff/VOCdevkit2007', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('output_dir', '/tmp3/jeff/test_img/result',
                           """Output Directory""")


def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label in zip(box_list, label_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)


def _viz_gt_bboxes(mc, images, bboxes, labels):
  for i in range(len(images)):
    # draw ground truth
    _draw_box(
        images[i], bboxes[i],
        [mc.CLASS_NAMES[idx] for idx in labels[i]],
        (0, 255, 0))

def test_read_batch():
  """Test read batch function"""
  assert FLAGS.dataset in ['KITTI', 'PASCAL_VOC', 'VID'], \
      'Either KITTI / PASCAL_VOC / VID'
  if FLAGS.dataset == 'KITTI':
    mc = kitti_vgg16_config()
    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
  elif FLAGS.dataset == 'PASCAL_VOC':
    mc = pascal_voc_vgg16_config()
    imdb = pascal_voc(FLAGS.image_set, FLAGS.year, FLAGS.data_path, mc)
  elif FLAGS.dataset == 'VID':
    mc = vid_vgg16_config()
    imdb = vid(FLAGS.image_set, FLAGS.data_path, mc)

  # read batch input
  image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
      bbox_per_batch = imdb.read_batch()
  joblib.dump(bbox_per_batch, '/tmp3/jeff/bbox.pkl')

  label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
      = [], [], [], [], []
  aidx_set = set()
  num_discarded_labels = 0
  num_labels = 0
  for i in range(len(label_per_batch)): # batch_size
    for j in range(len(label_per_batch[i])): # number of annotations
      num_labels += 1
      if (i, aidx_per_batch[i][j]) not in aidx_set:
        aidx_set.add((i, aidx_per_batch[i][j]))
        label_indices.append(
            [i, aidx_per_batch[i][j], label_per_batch[i][j]])
        mask_indices.append([i, aidx_per_batch[i][j]])
        bbox_indices.extend(
            [[i, aidx_per_batch[i][j], k] for k in range(4)])
        box_delta_values.extend(box_delta_per_batch[i][j])
        box_values.extend(bbox_per_batch[i][j])
      else:
        num_discarded_labels += 1

  print ('Warning: Discarded {}/({}) labels that are assigned to the same'
         'anchor'.format(num_discarded_labels, num_labels))

  # Visualize detections
  _viz_gt_bboxes(mc, image_per_batch, bbox_per_batch, label_per_batch)

  # Save the images
  for i,im in enumerate(image_per_batch):
    fname = os.path.join(FLAGS.output_dir, '{}.jpg'.format(i))
    cv2.imwrite(fname, im)

def main(argv=None):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  test_read_batch()


if __name__ == '__main__':
  tf.app.run()
