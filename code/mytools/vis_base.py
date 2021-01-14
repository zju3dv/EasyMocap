'''
  @ Date: 2020-11-28 17:23:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 17:11:51
  @ FilePath: /EasyMocap/code/mytools/vis_base.py
'''
import cv2
import numpy as np

import json
def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_colorbar(N = 1000, cmap = 'gist_rainbow'):
    from matplotlib import cm
    import numpy as np
    import random
    random.seed(666)
    
    cmaps = cm.get_cmap(cmap, N)
    x = np.linspace(0.0, 1.0, N)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = cm.get_cmap(cmap)(x)[:, :3]
    rgb = rgb[index, :]
    rgb = (rgb*255).astype(np.uint8).tolist()
    return rgb

colors_bar_rgb = generate_colorbar(cmap='hsv')
# colors_bar_rgb = read_json('config/colors.json')

def get_rgb(index):
    index = int(index)
    if index == -1:
        return (255, 255, 255)
    if index < -1:
        return (0, 0, 0)
    col = colors_bar_rgb[index%len(colors_bar_rgb)]
    # color = tuple([int(c*255) for c in col])
    return col

def plot_point(img, x, y, r, col, pid=-1):
    cv2.circle(img, (int(x+0.5), int(y+0.5)), r, col, -1)
    if pid != -1:
        cv2.putText(img, '{}'.format(pid), (int(x+0.5), int(y+0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)


def plot_line(img, pt1, pt2, lw, col):
    cv2.line(img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
        col, lw)

def plot_bbox(img, bbox, pid, vis_id=True):
    # 画bbox: (l, t, r, b)
    x1, y1, x2, y2 = bbox[:4]
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))
    color = get_rgb(pid)
    lw = max(img.shape[0]//300, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    if vis_id:
        cv2.putText(img, '{}'.format(pid), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def plot_keypoints(img, points, pid, config, vis_conf=False, use_limb_color=True, lw=2):
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= points.shape[0] or j >= points.shape[0]:
            continue
        pt1, pt2 = points[i], points[j]
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[2] > 0.01 and pt2[2] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
                col, lw)
    for i in range(len(points)):
        x, y, c = points[i]
        if c > 0.01:
            col = get_rgb(pid)
            cv2.circle(img, (int(x+0.5), int(y+0.5)), lw*2, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

def merge(images, row=-1, col=-1, resize=False, ret_range=False):
    if row == -1 and col == -1:
        from math import sqrt
        row = int(sqrt(len(images)) + 0.5)
        col = int(len(images)/ row + 0.5)
        if row > col:
            row, col = col, row
    if len(images) == 8:
        # basketball 场景
        row, col = 2, 4
        images = [images[i] for i in [0, 1, 2, 3, 7, 6, 5, 4]]
    if len(images) == 7:
        row, col = 3, 3
    height = images[0].shape[0]
    width = images[0].shape[1]
    ret_img = np.zeros((height * row, width * col, 3), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        scale = min(1000/ret_img.shape[0], 1800/ret_img.shape[1])
        while ret_img.shape[0] > 2000:
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img