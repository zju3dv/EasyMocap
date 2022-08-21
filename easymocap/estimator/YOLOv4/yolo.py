'''
  @ Date: 2020-12-10 16:39:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-04-21 23:53:40
  @ FilePath: /EasyMocapPublic/easymocap/estimator/YOLOv4/yolo.py
'''
from .darknet2pytorch import Darknet
import cv2
import torch
from os.path import join
import os
import numpy as np

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_processing(conf_thresh, nms_thresh, output):
    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for class person
        j = 0
        cls_argwhere = l_max_id == j
        ll_box_array = l_box_array[cls_argwhere, :]
        ll_max_conf = l_max_conf[cls_argwhere]
        ll_max_id = l_max_id[cls_argwhere]

        keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
        
        if (keep.size > 0):
            ll_box_array = ll_box_array[keep, :]
            ll_max_conf = ll_max_conf[keep]
            ll_max_id = ll_max_id[keep]
            bboxes = np.hstack([ll_box_array, ll_max_conf[:, None]])

        bboxes_batch.append(bboxes)

    return bboxes_batch

class YOLOv4:
    def __init__(self, device, ckpt_path, box_nms_thres, conf_thres,
        isWild=False) -> None:
        dirname = os.path.dirname(__file__)
        cfgfile = join(dirname, 'yolov4.cfg')
        namesfile = join(dirname, 'coco.names')
        self.model = Darknet(cfgfile)
        self.model.load_weights(ckpt_path)
        self.model.to(device)
        self.model.eval()
        class_names = load_class_names(namesfile)
        self.device = device
        self.box_nms_thres = box_nms_thres
        self.conf_thres = conf_thres
        self.isWild = isWild

    def predict_single(self, image):
        width  = image.shape[1]
        height = image.shape[0]
        tgt_width = self.model.width
        # 先缩小，再padding
        if width > height:
            tgt_shape = (tgt_width, int(height/width*tgt_width))
            resize = cv2.resize(image, tgt_shape)
            sized = np.zeros((tgt_width, tgt_width, 3), dtype=np.uint8)
            start = (sized.shape[0] - resize.shape[0])//2
            sized[start:start+resize.shape[0], :, :] = resize
            # pad_to_square
        elif width == height:
            sized = cv2.resize(image, (tgt_width, tgt_width))
            start = 0
        else:
            tgt_shape = (int(width/height*tgt_width), tgt_width)
            resize = cv2.resize(image, tgt_shape)
            sized = np.zeros((tgt_width, tgt_width, 3), dtype=np.uint8)
            start = (sized.shape[1] - resize.shape[1]) // 2
            sized[:, start:start+resize.shape[1], :] = resize
        img = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            output = self.model(img)
        bboxes = post_processing(self.conf_thres, self.box_nms_thres, output)[0]
        if len(bboxes) == 0:
            return bboxes
        if self.isWild:
            flag = ((bboxes[:, 2] - bboxes[:, 0]) < 0.8)&(((bboxes[:, 2] - bboxes[:, 0]) > 0.1)|((bboxes[:, 3] - bboxes[:, 1]) > 0.1))
            bboxes = bboxes[flag]
        if width >= height:
            bboxes[:, :4] *= width
            bboxes[:, 1] -= start*width/tgt_width
            bboxes[:, 3] -= start*width/tgt_width
        else:
            bboxes[:, :4] *= height
            bboxes[:, 0] -= start*height/tgt_width
            bboxes[:, 2] -= start*height/tgt_width
        # return bounding box
        return bboxes