import numpy as np
import cv2
import os
from os.path import join
from ..mytools import plot_cross, plot_line, plot_bbox, plot_keypoints, get_rgb, merge
from ..mytools.file_utils import get_bbox_from_pose
from ..dataset import CONFIG

# click and (start, end) is the output of the OpenCV callback
def vis_point(img, click, **kwargs):
    if click is not None:
        plot_cross(img, click[0], click[1], (255, 255, 255))
    return img

def vis_line(img, start, end, **kwargs):
    if start is not None and end is not None:
        lw = max(2, img.shape[0]//500)
        cv2.line(img, (int(start[0]), int(start[1])), 
            (int(end[0]), int(end[1])), (0, 255, 0), lw)
    return img

def vis_bbox(img, start, end, **kwargs):
    if start is not None and end is not None:
        lw = max(2, img.shape[0]//500)
        cv2.rectangle(img, (int(start[0]), int(start[1])), 
            (int(end[0]), int(end[1])), (0, 255, 0), lw)
    return img

def resize_to_screen(img, scale=1, **kwargs):
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def capture_screen(img, capture_screen=False, **kwargs):
    if capture_screen:
        from datetime import datetime
        time_now = datetime.now().strftime("%m-%d-%H:%M:%S")
        outname = join('capture', time_now+'.jpg')
        os.makedirs('capture', exist_ok=True)
        cv2.imwrite(outname, img)
        print('Capture current screen to {}'.format(outname))
    return img

def plot_text(img, annots, imgname, **kwargs):
    if 'isKeyframe' in annots.keys():
        if annots['isKeyframe']: # 关键帧使用红框表示
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), img.shape[1]//100)
        else: # 非关键帧使用绿框表示
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), img.shape[1]//100)
    imgname = '/'.join(imgname.split(os.sep)[-3:])
    text_size = int(max(1, img.shape[0]//1500))
    border = 20 * text_size
    width = 2 * text_size
    cv2.putText(img, '{}'.format(imgname), (border, img.shape[0]-border), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255), width)
    # 显示标注进度条:
    if 'frame' in kwargs.keys():
        width = img.shape[1]
        frame, nFrames = kwargs['frame'], kwargs['nFrames']
        lw = 12
        pos = lambda x: int(width*(x+1)/nFrames)
        COL_ALL = (0, 255, 0)
        COL_CUR = (255, 0, 0)
        COL_PIN = (255, 128, 128)
        plot_line(img, (0, lw/2), (width, lw/2), lw, COL_ALL)
        plot_line(img, (0, lw/2), (pos(frame), lw/2), lw, COL_CUR)
        top = pos(frame)
        pts = np.array([[top, lw], [top-lw, lw*4], [top+lw, lw*4]])
        cv2.fillPoly(img, [pts], COL_PIN)
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
        ratio = (y2-y1)/(x2-x1)
        w = 10*lw
        cv2.rectangle(img, 
            (int((x1+x2)/2-w), int((y1+y2)/2-w*ratio)), 
            (int((x1+x2)/2+w), int((y1+y2)/2+w*ratio)), 
            color, -1)
        cv2.putText(img, '{}'.format(pid), (int(x1), int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 2)
    return img

def plot_bbox_sp(img, annots, bbox_type='handl_bbox', add_center=False):
    assert bbox_type in ('bbox', 'bbox_handl2d', 'bbox_handr2d', 'bbox_face2d')
    for data in annots['annots']:
        if bbox_type not in data.keys():
            continue
        bbox = data[bbox_type]
        if bbox[-1] < 0.001: continue
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
        if add_center:
            cv2.rectangle(img,
                (int((x1+x2)/2-w), int((y1+y2)/2-w*ratio)),
                (int((x1+x2)/2+w), int((y1+y2)/2+w*ratio)),
                color, -1)
        cv2.putText(img, '{}'.format(pid), (int(x1), int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

def plot_bbox_factory(bbox_type, add_center=False):
    def ret_foo(img, annots, **kwargs):
        return plot_bbox_sp(img, annots, bbox_type=bbox_type, add_center=add_center)
    return ret_foo

def plot_skeleton(img, annots, body='body25', bbox_name='bbox', kpts_name='keypoints', **kwargs):
    annots = annots['annots']
    vis_conf = False
    for data in annots:
        pid = data.get('personID', -1)
        if kpts_name in data.keys():
            keypoints = data[kpts_name]
            plot_keypoints(img, keypoints, pid, CONFIG[body], vis_conf=vis_conf, use_limb_color=True)
        if bbox_name in data.keys():
            bbox = data[bbox_name]
            plot_bbox(img, bbox, pid)
        elif kpts_name in data.keys():
            bbox = get_bbox_from_pose(np.array(data[kpts_name]))
            plot_bbox(img, bbox, pid)
    return img

def plot_skeleton_factory(body):
    restore_key = {
        'body25': ('bbox', 'keypoints'),
        'handl': ('bbox_handl2d', 'handl2d'),
        'handr': ('bbox_handr2d', 'handr2d'),
        'face': ('bbox_face2d', 'face2d'),
    }
    bbox_name, kpts_name = restore_key[body]
    def ret_foo(img, annots, **kwargs):
        return plot_skeleton(img, annots, body, bbox_name, kpts_name)
    return ret_foo

def vis_active_bbox(img, annots, select, bbox_name, **kwargs):
    active = select[bbox_name]
    if active == -1 or active >= len(annots['annots']):
        return img
    else:
        bbox = annots['annots'][active][bbox_name]
        pid = annots['annots'][active]['personID']
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.rectangle(mask, 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            get_rgb(pid), -1)
        img = cv2.addWeighted(img, 0.6, mask, 0.4, 0)
    return img
