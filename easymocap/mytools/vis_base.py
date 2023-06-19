'''
  @ Date: 2020-11-28 17:23:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-27 15:13:56
  @ FilePath: /EasyMocapPublic/easymocap/mytools/vis_base.py
'''
import cv2
import numpy as np
import json

def generate_colorbar(N = 20, cmap = 'jet', rand=True, 
    ret_float=False, ret_array=False, ret_rgb=False):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    if rand:
        import random
        random.seed(666)
        index = [i for i in range(N)]
        random.shuffle(index)
        rgb = colorbar[index, :]
    else:
        rgb = colorbar
    if ret_rgb:
        rgb = rgb[:, ::-1]
    if ret_float:
        rgb = rgb/255.
    if not ret_array:
        rgb = rgb.tolist()
    return rgb

# colors_bar_rgb = generate_colorbar(cmap='hsv')
colors_bar_rgb = [
    (94, 124, 226), # 青色
    (255, 200, 87), # yellow
    (74,  189,  172), # green
    (8, 76, 97), # blue
    (219, 58, 52), # red
    (77, 40, 49), # brown
]

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    'g':[0,255/255,0],
    'k':[0,0,0],
    '_r':[255/255,0,0],
    '_g':[0,255/255,0],
    '_b':[0,0,255/255],
    '_k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'person': [255/255,255/255,255/255],
    'handl': [255/255,51/255,153/255],
    'handr': [51/255,255/255,153/255],
}

def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        # elif index == 0:
        #     return (245, 150, 150)
        col = list(colors_bar_rgb[index%len(colors_bar_rgb)])[::-1]
    elif isinstance(index, str):
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    else:
        raise TypeError('index should be int or str')
    return col

def get_rgb_01(index):
    col = get_rgb(index)
    return [i*1./255 for i in col[:3]]

def plot_point(img, x, y, r, col, pid=-1, font_scale=-1, circle_type=-1):
    cv2.circle(img, (int(x+0.5), int(y+0.5)), r, col, circle_type)
    if font_scale == -1:
        font_scale = img.shape[0]/4000
    if pid != -1:
        cv2.putText(img, '{}'.format(pid), (int(x+0.5), int(y+0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 1)


def plot_line(img, pt1, pt2, lw, col):
    cv2.line(img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
        col, lw)

def plot_cross(img, x, y, col, width=-1, lw=-1):
    if lw == -1:
        lw = max(1, int(round(img.shape[0]/1000)))
        width = lw * 5
    cv2.line(img, (int(x-width), int(y)), (int(x+width), int(y)), col, lw)
    cv2.line(img, (int(x), int(y-width)), (int(x), int(y+width)), col, lw)
    
def plot_bbox(img, bbox, pid, scale=1, vis_id=True):
    # 画bbox: (l, t, r, b)
    x1, y1, x2, y2, c = bbox
    if c < 0.01:return img
    x1 = int(round(x1*scale))
    x2 = int(round(x2*scale))
    y1 = int(round(y1*scale))
    y2 = int(round(y2*scale))
    color = get_rgb(pid)
    lw = max(img.shape[0]//300, 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    if vis_id:
        font_scale = img.shape[0]/1000
        cv2.putText(img, '{}'.format(pid), (x1, y1+int(25*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def plot_keypoints(img, points, pid, config, vis_conf=False, use_limb_color=True, lw=2, fliplr=False):
    lw = max(lw, 2)
    H, W = img.shape[:2]
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= len(points) or j >= len(points):
            continue
        if (i >25 or j > 25) and config['nJoints'] != 42:
            _lw = max(int(lw/4), 1)
        else:
            _lw = lw
        pt1, pt2 = points[i], points[j]
        if fliplr:
            pt1 = (W-pt1[0], pt1[1])
            pt2 = (W-pt2[0], pt2[1])
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]+0.5), int(pt1[1]+0.5)), (int(pt2[0]+0.5), int(pt2[1]+0.5)),
                col, _lw)
    for i in range(min(len(points), config['nJoints'])):
        x, y = points[i][0], points[i][1]
        if fliplr:
            x = W - x
        c = points[i][-1]
        if c > 0.01:
            text_size = img.shape[0]/1000
            col = get_rgb(pid)
            radius = int(lw/1.5)
            if i > 25 and config['nJoints'] != 42:
                radius = max(int(radius/4), 1)
            cv2.circle(img, (int(x+0.5), int(y+0.5)), radius, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, text_size, col, 2)

def plot_keypoints_auto(img, points, pid, vis_conf=False, use_limb_color=True, scale=1, lw=-1, config_name=None, lw_factor=1):
    from ..dataset.config import CONFIG
    if config_name is None:
        config_name = {25: 'body25', 15: 'body15', 21: 'hand', 42:'handlr', 17: 'coco', 1:'points', 67:'bodyhand', 137: 'total', 79:'up',
            19:'ochuman'}[len(points)]
    config = CONFIG[config_name]
    if lw == -1:
        lw = img.shape[0]//200
    if config_name == 'hand':
        lw = img.shape[0]//100
    lw = max(lw, 1)
    for ii, (i, j) in enumerate(config['kintree']):
        if i >= len(points) or j >= len(points):
            continue
        if i >= 25 and config_name in ['bodyhand', 'total']:
            lw = max(img.shape[0]//400, 1)
        pt1, pt2 = points[i], points[j]
        if use_limb_color:
            col = get_rgb(config['colors'][ii])
        else:
            col = get_rgb(pid)
        if pt1[0] < -10000 or pt1[1] < -10000 or pt1[0] > 10000 or pt1[1] > 10000:
            continue
        if pt2[0] < -10000 or pt2[1] < -10000 or pt2[0] > 10000 or pt2[1] > 10000:
            continue
        if pt1[-1] > 0.01 and pt2[-1] > 0.01:
            image = cv2.line(
                img, (int(pt1[0]*scale+0.5), int(pt1[1]*scale+0.5)), (int(pt2[0]*scale+0.5), int(pt2[1]*scale+0.5)),
                col, lw)
    lw = img.shape[0]//200
    if config_name == 'hand':
        lw = img.shape[0]//500
    lw = max(lw, 1)
    for i in range(len(points)):
        x, y = points[i][0]*scale, points[i][1]*scale
        if x < 0 or y < 0 or x >10000 or y >10000:
            continue
        if i >= 25 and config_name in ['bodyhand', 'total']:
            lw = max(img.shape[0]//400, 1)
        c = points[i][-1]
        if c > 0.01:
            col = get_rgb(pid)
            if len(points) == 1:
                _lw = max(0, int(lw * lw_factor))
                cv2.circle(img, (int(x+0.5), int(y+0.5)), _lw*2, col, lw*2)
                plot_cross(img, int(x+0.5), int(y+0.5), width=_lw, col=col, lw=lw*2)
            else:
                cv2.circle(img, (int(x+0.5), int(y+0.5)), lw*2, col, -1)
            if vis_conf:
                cv2.putText(img, '{:.1f}'.format(c), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

def plot_keypoints_total(img, annots, scale, pid_offset=0):
    _lw = img.shape[0] // 150
    for annot in annots:
        pid = annot['personID'] + pid_offset
        for key in ['keypoints', 'handl2d', 'handr2d']:
            if key not in annot.keys():continue
            if key in ['handl2d', 'handr2d', 'face2d']:
                lw = _lw // 2
            else:
                lw = _lw
            lw = max(lw, 1)
            plot_keypoints_auto(img, annot[key], pid, vis_conf=False, use_limb_color=False, scale=scale, lw=lw)
            if 'bbox' not in annot.keys() or (annot['bbox'][0] < 0 or annot['bbox'][0] >10000):
                continue
            plot_bbox(img, annot['bbox'], pid, scale=scale, vis_id=True)
    return img

def plot_points2d(img, points2d, lines, lw=-1, col=(0, 255, 0), putText=True, style='+'):
    # 将2d点画上去
    if points2d.shape[1] == 2:
        points2d = np.hstack([points2d, np.ones((points2d.shape[0], 1))])
    if lw == -1:
        lw = img.shape[0]//200
    for i, (x, y, v) in enumerate(points2d):
        if v < 0.01:
            continue
        c = col
        if '+' in style:
            plot_cross(img, x, y, width=10, col=c, lw=lw*2)
        if 'o' in style:
            cv2.circle(img, (int(x), int(y)), 10, c, lw*2)
        cv2.circle(img, (int(x), int(y)), lw, c, lw)
        if putText:
            c = col[::-1]
            font_scale = img.shape[0]/1000
            cv2.putText(img, '{}'.format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, 2)
    for i, j in lines:
        if points2d[i][2] < 0.01 or points2d[j][2] < 0.01:
            continue
        plot_line(img, points2d[i], points2d[j], max(1, lw//2), col)

row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}

row_col_square = {
    2: (2, 1),
    7: (3, 3),
    8: (3, 3),
    9: (3, 3),
    26: (5, 5)
}

def get_row_col(l, square):
    if square and l in row_col_square.keys():
        return row_col_square[l]
    if l in row_col_.keys():
        return row_col_[l]
    else:
        from math import sqrt
        row = int(sqrt(l) + 0.5)
        col = int(l/ row + 0.5)
        if row*col<l:
            col = col + 1
        if row > col:
            row, col = col, row
        return row, col

def merge(images, row=-1, col=-1, resize=False, ret_range=False, square=False, **kwargs):
    if row == -1 and col == -1:
        row, col = get_row_col(len(images), square)
    height = images[0].shape[0]
    width = images[0].shape[1]
    # special case
    if height > width:
        if len(images) == 3:
            row, col = 1, 3
    if len(images[0].shape) > 2:
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
    else:
        ret_img = np.zeros((height * row, width * col), dtype=np.uint8) + 255
    ranges = []
    for i in range(row):
        for j in range(col):
            if i*col + j >= len(images):
                break
            img = images[i * col + j]
            # resize the image size
            img = cv2.resize(img, (width, height))
            ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
            ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
    if resize:
        min_height = 1000
        if ret_img.shape[0] > min_height:
            scale = min_height/ret_img.shape[0]
            ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
    if ret_range:
        return ret_img, ranges
    return ret_img