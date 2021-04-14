import numpy as np
import cv2
import os
from os.path import join
from ..mytools import plot_cross, plot_line, plot_bbox, plot_keypoints, get_rgb
from ..dataset import CONFIG

# click and (start, end) is the output of the OpenCV callback
def vis_point(img, click, **kwargs):
    if click is not None:
        plot_cross(img, click[0], click[1], (255, 255, 255))
    return img

def vis_line(img, start, end, **kwargs):
    if start is not None and end is not None:
        cv2.line(img, (int(start[0]), int(start[1])), 
            (int(end[0]), int(end[1])), (0, 255, 0), 1)
    return img

def resize_to_screen(img, scale=1, capture_screen=False, **kwargs):
    if capture_screen:
        from datetime import datetime
        time_now = datetime.now().strftime("%m-%d-%H:%M:%S")
        outname = join('capture', time_now+'.jpg')
        os.makedirs('capture', exist_ok=True)
        cv2.imwrite(outname, img)
        print('Capture current screen to {}'.format(outname))
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def plot_text(img, annots, **kwargs):
    if annots['isKeyframe']: # 关键帧使用红框表示
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), img.shape[1]//100)
    else: # 非关键帧使用绿框表示
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), img.shape[1]//100)
    text_size = int(max(1, img.shape[0]//1500))
    border = 20 * text_size
    width = 2 * text_size
    cv2.putText(img, '{}'.format(annots['filename']), (border, img.shape[0]-border), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), width)
    return img

def plot_bbox_body(img, annots, **kwargs):
    annots = annots['annots']
    for data in annots:
        bbox = data['bbox']
        # 画一个X形
        x1, y1, x2, y2 = bbox[:4]
        pid = data['personID']
        color = get_rgb(pid)
        lw = max(1, int((x2 - x1)//100))
        plot_line(img, (x1, y1), (x2, y2), lw, color)
        plot_line(img, (x1, y2), (x2, y1), lw, color)
        # border
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, lw+1)
        ratio = (y2-y1)/(x2-x1)/2
        w = 10*lw
        cv2.rectangle(img, 
            (int((x1+x2)/2-w), int((y1+y2)/2-w*ratio)), 
            (int((x1+x2)/2+w), int((y1+y2)/2+w*ratio)), 
            color, -1)
        cv2.putText(img, '{}'.format(pid), (int(x1), int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def plot_skeleton(img, annots, **kwargs):
    annots = annots['annots']
    vis_conf = False
    for data in annots:
        bbox, keypoints = data['bbox'], data['keypoints']
        if False:
            pid = data.get('matchID', -1)
        else:
            pid = data.get('personID', -1)
        plot_bbox(img, bbox, pid)
        if True:
            plot_keypoints(img, keypoints, pid, CONFIG['body25'], vis_conf=vis_conf, use_limb_color=True)
            if 'handl2d' in data.keys():
                plot_keypoints(img, data['handl2d'], pid, CONFIG['hand'], vis_conf=vis_conf, lw=1, use_limb_color=False)
                plot_keypoints(img, data['handr2d'], pid, CONFIG['hand'], vis_conf=vis_conf, lw=1, use_limb_color=False)
                plot_keypoints(img, data['face2d'], pid, CONFIG['face'], vis_conf=vis_conf, lw=1, use_limb_color=False)
    return img

def plot_keypoints_whole(img, points, kintree):
    for ii, (i, j) in enumerate(kintree):
        if i >= len(points) or j >= len(points):
            continue
        col = (255, 240, 160)
        lw = 4
        pt1, pt2 = points[i], points[j]
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
                col, lw)

def plot_skeleton_simple(img, annots, **kwargs):
    annots = annots['annots']
    vis_conf = False
    for data in annots:
        bbox, keypoints = data['bbox'], data['keypoints']
        pid = data.get('personID', -1)
        plot_keypoints_whole(img, keypoints, CONFIG['body25']['kintree'])
    return img

def vis_active_bbox(img, annots, select, **kwargs):
    active = select['bbox']
    if active == -1:
        return img
    else:
        bbox = annots['annots'][active]['bbox']
        pid = annots['annots'][active]['personID']
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.rectangle(mask, 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            get_rgb(pid), -1)
        img = cv2.addWeighted(img, 0.6, mask, 0.4, 0)
    return img
