# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for pascal voc"""

import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET

from utils.util import bbox_transform_inv
from dataset.imdb import imdb
from dataset.voc_eval import voc_eval

class pascal_voc(imdb):
  def __init__(self, image_set, year, data_path, mc):
    imdb.__init__(self, 'voc_'+year+'_'+image_set, mc)
    self._year = year
    self._image_set = image_set
    self._data_root_path = data_path
    self._data_path = os.path.join(self._data_root_path, 'VOC' + self._year)
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_pascal_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._data_path, 'JPEGImages', idx+'.jpg')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_pascal_annotation(self):
    idx2annotation = {}
    for index in self._image_idx:
      filename = os.path.join(self._data_path, 'Annotations', index+'.xml')
      tree = ET.parse(filename)
      objs = tree.findall('object')
      objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
      bboxes = []
      for obj in objs:
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        xmin = float(bbox.find('xmin').text) - 1
        xmax = float(bbox.find('xmax').text) - 1
        ymin = float(bbox.find('ymin').text) - 1
        ymax = float(bbox.find('ymax').text) - 1
        assert xmin >= 0.0 and xmin <= xmax, \
            'Invalid bounding box x-coord xmin {} or xmax {} at {}.xml' \
                .format(xmin, xmax, index)
        assert ymin >= 0.0 and ymin <= ymax, \
            'Invalid bounding box y-coord ymin {} or ymax {} at {}.xml' \
                .format(ymin, ymax, index)
        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        cls = self._class_to_idx[obj.find('name').text.lower().strip()]
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

    for cls_idx, cls in enumerate(self._classes):
      det_file_name = det_file_path_template.format(cls)
      with open(det_file_name, 'wt') as f:
        for im_idx, index in enumerate(self._image_idx):
          dets = all_boxes[cls_idx][im_idx]
          # VOC expects 1-based indices
          for k in xrange(len(dets)):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(index, dets[k][-1], 
                       dets[k][0]+1, dets[k][1]+1,
                       dets[k][2]+1, dets[k][3]+1)
            )

    # Evaluate detection results
    annopath = os.path.join(
        self._data_root_path,
        'VOC'+self._year,
        'Annotations',
        '{:s}.xml'
    )
    imagesetfile = os.path.join(
        self._data_root_path,
        'VOC'+self._year,
        'ImageSets',
        'Main',
        self._image_set+'.txt'
    )
    cachedir = os.path.join(self._data_root_path, 'annotations_cache')
    aps = []
    use_07_metric = True if int(self._year) < 2010 else False
    for i, cls in enumerate(self._classes):
      filename = det_file_path_template.format(cls)
      _,  _, ap = voc_eval(
          filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
          use_07_metric=use_07_metric)
      aps += [ap]
      print ('{:s}: AP = {:.4f}'.format(cls, ap))

    print ('Mean AP = {:.4f}'.format(np.mean(aps)))
    return aps, self._classes

  def do_detection_analysis_in_eval(self, eval_dir, global_step):
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    det_error_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step),
        'error_analysis')
    if not os.path.exists(det_error_dir):
      os.makedirs(det_error_dir)
    det_error_file = os.path.join(det_error_dir, 'det_error_file.txt')

    stats = self.analyze_detections(det_file_dir, det_error_file)
    ims = self.visualize_detections(
        image_dir=self._image_path,
        image_format='.png',
        det_error_file=det_error_file,
        output_image_dir=det_error_dir,
        num_det_per_type=10
    )

    return stats, ims

  def analyze_detections(self, detection_file_dir, det_error_file):
    def _save_detection(f, idx, error_type, det, score):
      f.write(
          '{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:.3f}\n'.format(
              idx, error_type,
              det[0]-det[2]/2., det[1]-det[3]/2.,
              det[0]+det[2]/2., det[1]+det[3]/2.,
              self._classes[int(det[4])], 
              score
          )
      )

    # load detections
    self._det_rois = {}
    for idx in self._image_idx:
      det_file_name = os.path.join(detection_file_dir, idx+'.txt')
      with open(det_file_name) as f:
        lines = f.readlines()
      f.close()
      bboxes = []
      for line in lines:
        obj = line.strip().split(' ')
        cls = self._class_to_idx[obj[0].lower().strip()]
        xmin = float(obj[4])
        ymin = float(obj[5])
        xmax = float(obj[6])
        ymax = float(obj[7])
        score = float(obj[-1])

        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        bboxes.append([x, y, w, h, cls, score])
      bboxes.sort(key=lambda x: x[-1], reverse=True)
      self._det_rois[idx] = bboxes

    # do error analysis
    num_objs = 0.
    num_dets = 0.
    num_correct = 0.
    num_loc_error = 0.
    num_cls_error = 0.
    num_bg_error = 0.
    num_repeated_error = 0.
    num_detected_obj = 0.

    with open(det_error_file, 'w') as f:
      for idx in self._image_idx:
        gt_bboxes = np.array(self._rois[idx])
        num_objs += len(gt_bboxes)
        detected = [False]*len(gt_bboxes)

        det_bboxes = self._det_rois[idx]
        for i, det in enumerate(det_bboxes):
          if i < len(gt_bboxes):
            num_dets += 1
          ious = batch_iou(gt_bboxes[:, :4], det[:4])
          max_iou = np.max(ious)
          gt_idx = np.argmax(ious)
          # if not detected[gt_idx]:
          #   if max_iou > 0.1:
          #     if gt_bboxes[gt_idx, 4] == det[4]:
          #       if max_iou >= 0.5:
          #         if i < len(gt_bboxes):
          #           num_correct += 1
          #         detected[gt_idx] = True
          #       else:
          #         if i < len(gt_bboxes):
          #           num_loc_error += 1
          #           _save_detection(f, idx, 'loc', det, det[5])
          #     else:
          #       if i < len(gt_bboxes):
          #         num_cls_error += 1
          #         _save_detection(f, idx, 'cls', det, det[5])
          #   else:
          #     if i < len(gt_bboxes):
          #       num_bg_error += 1
          #       _save_detection(f, idx, 'bg', det, det[5])
          # else:
          #   if i < len(gt_bboxes):
          #     num_repeated_error += 1

          if max_iou > 0.1:
            if gt_bboxes[gt_idx, 4] == det[4]:
              if max_iou >= 0.5:
                if i < len(gt_bboxes):
                  if not detected[gt_idx]:
                    num_correct += 1
                    detected[gt_idx] = True
                  else:
                    num_repeated_error += 1
              else:
                if i < len(gt_bboxes):
                  num_loc_error += 1
                  _save_detection(f, idx, 'loc', det, det[5])
            else:
              if i < len(gt_bboxes):
                num_cls_error += 1
                _save_detection(f, idx, 'cls', det, det[5])
          else:
            if i < len(gt_bboxes):
              num_bg_error += 1
              _save_detection(f, idx, 'bg', det, det[5])

        for i, gt in enumerate(gt_bboxes):
          if not detected[i]:
            _save_detection(f, idx, 'missed', gt, -1.0)
        num_detected_obj += sum(detected)
    f.close()

    print ('Detection Analysis:')
    print ('    Number of detections: {}'.format(num_dets))
    print ('    Number of objects: {}'.format(num_objs))
    print ('    Percentage of correct detections: {}'.format(
      num_correct/num_dets))
    print ('    Percentage of localization error: {}'.format(
      num_loc_error/num_dets))
    print ('    Percentage of classification error: {}'.format(
      num_cls_error/num_dets))
    print ('    Percentage of background error: {}'.format(
      num_bg_error/num_dets))
    print ('    Percentage of repeated detections: {}'.format(
      num_repeated_error/num_dets))
    print ('    Recall: {}'.format(
      num_detected_obj/num_objs))

    out = {}
    out['num of detections'] = num_dets
    out['num of objects'] = num_objs
    out['% correct detections'] = num_correct/num_dets
    out['% localization error'] = num_loc_error/num_dets
    out['% classification error'] = num_cls_error/num_dets
    out['% background error'] = num_bg_error/num_dets
    out['% repeated error'] = num_repeated_error/num_dets
    out['% recall'] = num_detected_obj/num_objs

    return out
