from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

import numpy as np
import torch
import json
import os

import torch.utils.data as data

class shipcoco(data.Dataset):
  num_classes = 2
  default_resolution = [512, 512]
  # mean = np.array([0.314, 0.325, 0.290],
  mean = np.array([0.408, 0.447, 0.470],
                   dtype=np.float32).reshape(1, 1, 3)
  # std  = np.array([0.185, 0.194, 0.215],
  std  = np.array([0.289, 0.274, 0.278],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(shipcoco, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'shipcoco')
    self.img_dir = os.path.join(self.data_dir, 'images')
    _ann_name = {'train': 'train', 'val': 'val'}
    self.annot_path = os.path.join(
      self.data_dir, 'annotations', 
      'shipcoco_{}.json').format(_ann_name[split])
    self.max_objs = 50
    self.class_name = ['__background__', "ship", "carrier"]
    # self.class_name = ['__background__', "ship", "carrier", "submarine"]
    self._valid_ids = np.arange(1, 3, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing pascal {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))


  def __len__(self):
    return self.num_samples




  def convert_eval_format(self, all_bboxes):
    categories = ['ship','aircraft']
    f = open('../data/shipcoco/annotations/shipcoco_val.json','r')
    val_json = json.load(f)
    for image_id in all_bboxes:
      output_path = '../mAP-master/input/detection-results/' + val_json['images'][image_id-1]['file_name'][:9]+'txt'
      file_out = open(output_path,'w')
      flag = 0
      for cls_ind in all_bboxes[image_id]:
        for bbox in all_bboxes[image_id][cls_ind]:
          if bbox[4] > self.opt.vis_thresh:
            if flag == 0:
              flag = 1
              file_out.write(categories[cls_ind - 1] + " {:.2f} ".format(bbox[4]) + str(round(bbox[0])) + ' ' + str(round(bbox[1])) + ' ' + str(round(bbox[2])) + ' ' + str(round(bbox[3])))
            else:
              file_out.write('\n' + categories[cls_ind - 1] + " {:.2f} ".format(bbox[4]) + str(round(bbox[0])) + ' ' + str(round(bbox[1])) + ' ' + str(round(bbox[2])) + ' ' + str(round(bbox[3])))
      file_out.close()
    f.close()

  def run_eval(self, results, save_dir):
    self.convert_eval_format(results)



