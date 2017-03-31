
"""Image data base class for ILSVRC 2013"""

import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET

from dataset.imdb import imdb

class imagenet(imdb):
  def __init__(self, image_set, data_path, mc):
    imdb.__init__(self, 'ILSVRC2013_'+image_set, mc)
    self._image_set = image_set
    self._data_path = data_path
    self._classes = self.mc.CLASS_NAMES
    #self._class_to_idx = dict(zip(self.classes, range(self.num_classes)))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 

    # a list of class labels
    self.labels = self._load_imagenet_labels()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(self._data_path, 'ImageSets',
                                  self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._data_path, 'Data', self._image_set, idx+'.JPEG')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_imagenet_labels(self):
    labels_file = os.path.join(self._data_path, 'ILSVRC2013_devkit', \
                               'data', 'ILSVRC2013_clsloc_validation_ground_truth.txt')
    assert os.path.exists(labels_file), \
        'File does not exist: {}'.format(labels_file)
    with open(labels_file, 'r') as f:
      lines = f.readlines()
      labels = [ int(line.strip()) for line in lines ] 
    return labels

  def read_cls_batch(self, shuffle=True):
    """Read a batch of images and labels
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      images: list of arrays [h, w, c]
      labels: list of class indexes
      scales: list of resize scale factor
    """
    mc = self.mc
    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    images, labels, scales = [], [], []
    for i in batch_idx:
      im = cv2.imread(self._image_path_at(i))
      if mc.SUB_BGR_MEANS:
        im = im.astype(np.float32, copy=False)
        im -= mc.BGR_MEANS
      orig_h, orig_w, _ = [float(v) for v in im.shape]
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
      x_scale = mc.IMAGE_WIDTH/orig_w
      y_scale = mc.IMAGE_HEIGHT/orig_h
      images.append(im)
      label_idx = int(i.split('_')[-1])-1
      labels.append(self.labels[label_idx]-1)
      scales.append((x_scale, y_scale))

    return images, labels, scales
