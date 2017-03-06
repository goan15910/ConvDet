
"""Image data base class for vid"""

import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET

from utils.util import bbox_transform_inv
from dataset.imdb import imdb
from dataset.vid_eval import vid_eval

class vid(imdb):
  def __init__(self, image_set, data_path, mc):
    imdb.__init__(self, 'vid_'+image_set, mc)
    self._image_set = image_set
    self._data_root_path = data_path
    self._data_path = os.path.join(data_path, 'Data/VID', image_set)
    self._idx_path = os.path.join(data_path, 'ImageSets/VID')
    self._anno_path = os.path.join(data_path, 'Annotations/VID', image_set)
    self._classes = self.mc.CLASS_NAMES
    self._raw_cnames = ('n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061',
                     'n02924116', 'n02958343', 'n02402425', 'n02084071', 'n02121808',
                     'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451',
                     'n02129165', 'n01674464', 'n02484322', 'n03790512', 'n02324045',
                     'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604',
                     'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049')
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))
    self._raw_cname_to_idx = dict(zip(self._raw_cnames, xrange(len(self._raw_cnames))))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_vid_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(self._idx_path, \
                                  self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._data_path, idx+'.JPEG')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_vid_annotation(self):
    idx2annotation = {}
    for index in self._image_idx:
      filename = os.path.join(self._anno_path, index+'.xml')
      tree = ET.parse(filename)
      objs = tree.findall('object')
      bboxes = []
      for obj in objs:
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)
        assert xmin >= 0.0 and xmin <= xmax, \
            'Invalid bounding box x-coord xmin {} or xmax {} at {}.xml' \
                .format(xmin, xmax, index)
        assert ymin >= 0.0 and ymin <= ymax, \
            'Invalid bounding box y-coord ymin {} or ymax {} at {}.xml' \
                .format(ymin, ymax, index)
        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        cls = self._raw_cname_to_idx[obj.find('name').text.lower().strip()]
        bboxes.append([x, y, w, h, cls])

      idx2annotation[index] = bboxes

    return idx2annotation

  def evaluate_detections(self, eval_dir, global_step, all_boxes):
    """Evaluate detection results.
    Args:
      eval_dir: directory to write evaluation logs
      global_step: step of the checkpoint
      all_boxes: all_boxes[cls][image] = N x 5 arrays of 
        [xmin, ymin, xmax, ymax, score]
    Returns:
      aps: array of average precisions.
      names: class names corresponding to each ap
    """
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step))
    if not os.path.isdir(det_file_dir):
      os.mkdir(det_file_dir)
    det_file_path_template = os.path.join(det_file_dir, '{:s}.txt')

    #TODO(jeff): customize VID cls
    for cls_idx, cls in enumerate(self._classes):
      det_file_name = det_file_path_template.format(cls)
      with open(det_file_name, 'wt') as f:
        for im_idx, index in enumerate(self._image_idx):
          dets = all_boxes[cls_idx][im_idx]
          #TODO(jeff): customize VID indices
          for k in xrange(len(dets)):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(index, dets[k][-1], 
                       dets[k][0], dets[k][1],
                       dets[k][2], dets[k][3])
            )

    # Evaluate detection results
    #TODO(jeff): customize annopath & imagesetfile
    #annopath = 
    #imagesetfile = 
    cachedir = os.path.join(self._data_root_path, 'annotations_cache')
    #TODO(jeff): customize AP evaluation
    aps = []
    for i, cls in enumerate(self._classes):
      filename = det_file_path_template.format(cls)
      _,  _, ap = vid_eval(
          filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
      aps += [ap]
      print ('{:s}: AP = {:.4f}'.format(cls, ap))

    print ('Mean AP = {:.4f}'.format(np.mean(aps)))
    return aps, self._classes
