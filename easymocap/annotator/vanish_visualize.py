'''
  @ Date: 2021-07-13 21:12:15
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-13 21:12:46
  @ FilePath: /EasyMocap/easymocap/annotator/vanish_visualize.py
'''
import cv2
import numpy as np
from .basic_visualize import plot_cross

def vis_vanish_lines(img, annots, **kwargs):
    if 'vanish_line' not in annots.keys():
        annots['vanish_line'] = [[], [], []]
    if 'vanish_point' not in annots.keys():
        annots['vanish_point'] = [[], [], []]
    colors = [(96, 96, 255), (96, 255, 96), (255, 64, 64)]

    for i in range(3):
        point = annots['vanish_point'][i]
        if len(point) == 0:
            continue
        x, y, c = point
        plot_cross(img, x, y, colors[i])
        points = np.array(annots['vanish_line'][i]).reshape(-1, 3)
        for (xx, yy, conf) in points:
            plot_cross(img, xx, yy, col=colors[i])
            cv2.line(img, (int(x), int(y)), (int(xx), int(yy)), colors[i], 2)

    for i in range(3):
        for pt1, pt2 in annots['vanish_line'][i]:
            cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), colors[i], 2)

    return img

