from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import sys
import random

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from PIL import Image

sys.path.append("../..")
from yolov3.utils import *

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt.model_def, opt.train_branch)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results, image_or_path_or_tensor):
   raise NotImplementedError


  def IoU(self, box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

  def PBS(self, center_outputs, yolo_outputs):
    thres_center = 1
    thres_yolo = 1
    thres_iou = 0.3
    # print(center_outputs)
    center_outputs_0 = torch.from_numpy(center_outputs[1])
    center_outputs_1 = torch.from_numpy(center_outputs[2])

    tmp_0 = torch.zeros(center_outputs_0.size(0),1)
    tmp_1 = torch.ones(center_outputs_1.size(0),1)
    tmp_0 = torch.cat((center_outputs_0, tmp_0), 1)
    tmp_1 = torch.cat((center_outputs_1, tmp_1), 1)
    center_outputs = torch.cat((tmp_0, tmp_1),0)
    center_outputs = center_outputs[center_outputs[:,4] > self.opt.vis_thresh]
    # print(center_outputs)

    keep_boxes_center = []
    tmp_boxes_center = []
    invalid_yolo = torch.zeros(yolo_outputs.size(0), dtype=torch.uint8)
    while center_outputs.size(0):
      large_overlap = self.IoU(center_outputs[0, :4].unsqueeze(0), yolo_outputs[:, :4]) > thres_iou
      label_match = center_outputs[0, -1] == yolo_outputs[:, -1]
      invalid_yolo_tmp = large_overlap & label_match

      if invalid_yolo_tmp.size(0)==0:
        tmp_boxes_center += [center_outputs[0]]
      else:
        keep_boxes_center += [center_outputs[0]]
        invalid_yolo = invalid_yolo | invalid_yolo_tmp
      center_outputs = center_outputs[1:,:]
    yolo_outputs = yolo_outputs[~invalid_yolo]

    keep_boxes_center = torch.tensor(keep_boxes_center).reshape(-1,6) if keep_boxes_center == [] else torch.stack(keep_boxes_center)
    tmp_boxes_center = torch.tensor(tmp_boxes_center).reshape(-1,6) if tmp_boxes_center == [] else torch.stack(tmp_boxes_center)
    yolo_outputs = yolo_outputs[yolo_outputs[:,4]>thres_yolo]
    tmp_boxes_center = tmp_boxes_center[tmp_boxes_center[:,4]>thres_center]

    pbs_outputs = torch.cat((keep_boxes_center,tmp_boxes_center,yolo_outputs),0)

    return pbs_outputs

  def show_results_pbs(self, debugger, image, results, image_or_path_or_tensor):
    #---------------------results-------------------------
    results = results.numpy()

    categories = ['ship','aircraft']
    image_name = image_or_path_or_tensor.split('/')[-1].split('.')[0]+'.txt'
    output_path = '../mAP-master/input/detection-results/pbs/' + image_name
    file_out = open(output_path,'w')
    flag = 0
    #-----------------------------------------------------
    debugger.add_img(image, img_id='ctdet')
    for bbox in results:
      debugger.add_coco_bbox(bbox[:4], bbox[-1], bbox[4], img_id='ctdet')
      #--------------------------results------------
      if flag == 0:
        flag = 1
        file_out.write(categories[int(bbox[-1])] + " {:.2f} ".format(bbox[4]) + str(int(round(bbox[0]))) + ' ' + str(int(round(bbox[1]))) + ' ' + str(int(round(bbox[2]))) + ' ' + str(int(round(bbox[3]))))
      else:
        file_out.write('\n' + categories[int(bbox[-1])] + " {:.2f} ".format(bbox[4]) + str(int(round(bbox[0]))) + ' ' + str(int(round(bbox[1]))) + ' ' + str(int(round(bbox[2]))) + ' ' + str(int(round(bbox[3]))))
    file_out.close()
          #---------------------------------------------


  def show_results_yolo(self, debugger, image, results, image_or_path_or_tensor):
    #---------------------results-------------------------
    results = results.numpy()
    # print(results)
    categories = ['ship','aircraft']
    image_name = image_or_path_or_tensor.split('/')[-1].split('.')[0]+'.txt'
    output_path = '../mAP-master/input/detection-results/yolo/' + image_name

    file_out = open(output_path,'w')
    flag = 0
    #-----------------------------------------------------
    debugger.add_img(image, img_id='ctdet')
    for bbox in results:
      debugger.add_coco_bbox(bbox[:4], bbox[-1], bbox[4], img_id='ctdet')
      #--------------------------results------------
      if flag == 0:
        flag = 1
        file_out.write(categories[int(bbox[-1])] + " {:.2f} ".format(bbox[4]) + str(int(round(bbox[0]))) + ' ' + str(int(round(bbox[1]))) + ' ' + str(int(round(bbox[2]))) + ' ' + str(int(round(bbox[3]))))
      else:
        file_out.write('\n' + categories[int(bbox[-1])] + " {:.2f} ".format(bbox[4]) + str(int(round(bbox[0]))) + ' ' + str(int(round(bbox[1]))) + ' ' + str(int(round(bbox[2]))) + ' ' + str(int(round(bbox[3]))))
    file_out.close()
          #---------------------------------------------
    # debugger.save_all_imgs(path='/home/zhyj/document/CenterNet-master/outputs/Yolo', genID=True)

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    yolo_detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, yolo_dets, forward_time = self.process(images, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)

      yolo_detections.append(yolo_dets)

    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time



    img_yolo = np.array(Image.open(image_or_path_or_tensor))
    yolo_outputs = yolo_detections[0][0]
    if yolo_outputs is not None:
        yolo_outputs = rescale_boxes(yolo_outputs, 512, img_yolo.shape[:2])
    yolo_outputs = yolo_outputs[:,(0,1,2,3,4,-1)] if not yolo_outputs is None else torch.tensor([]).reshape(-1,6)


    if self.opt.results_to_save=='center':
      self.show_results(debugger, image, results, image_or_path_or_tensor)
    elif self.opt.results_to_save=='yolo':
      self.show_results_yolo(debugger, image, yolo_outputs, image_or_path_or_tensor)

    else:

      #-----------------PBS------------------
      pbs_outputs = self.PBS(results, yolo_outputs)
      self.show_results_pbs(debugger, image, pbs_outputs, image_or_path_or_tensor)

    #--------------------------------------
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}


