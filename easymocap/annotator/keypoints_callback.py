'''
  @ Date: 2021-04-22 11:40:31
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-10 16:00:15
  @ FilePath: /EasyMocap/easymocap/annotator/keypoints_callback.py
'''
import numpy as np
from .bbox_callback import findNearestPoint

def callback_select_joints(start, end, annots, select, bbox_name='bbox', kpts_name='keypoints', **kwargs):
    if start is None or end is None:
        select['joints'] = -1
        return 0
    if start[0] == end[0] and start[1] == end[1]:
        select['joints'] = -1
        return 0
    if select['corner'] != -1:
        return 0
    # 判断选择了哪个角点
    annots = annots['annots']
    # not select a bbox
    if select[bbox_name] == -1 and select['joints'] == -1:
        corners = []
        for annot in annots:
            corners.append(np.array(annot[kpts_name]))
        corners = np.stack(corners)
        flag, minid = findNearestPoint(corners[..., :2], start)
        if flag:
            select[bbox_name] = minid[0]
            select['joints'] = minid[1]
        else:
            select['joints'] = -1
    # have selected a bbox, not select a corner
    elif select[bbox_name] != -1 and select['joints'] == -1:
        i = select[bbox_name]
        corners = np.array(annots[i][kpts_name])[:, :2]
        flag, minid = findNearestPoint(corners, start)
        if flag:
            select['joints'] = minid[0]
    # have selected a bbox, and select a corner
    elif select[bbox_name] != -1 and select['joints'] != -1:
        x, y = end
        # Move the corner
        data = annots[select[bbox_name]]
        nj = select['joints']
        data[kpts_name][nj][0] = x
        data[kpts_name][nj][1] = y
        if kpts_name == 'keypoints': # for body
            if nj in [1, 8]:
                return 0
            if nj in [2, 5]:
                data[kpts_name][1][0] = (data[kpts_name][2][0] + data[kpts_name][5][0])/2
                data[kpts_name][1][1] = (data[kpts_name][2][1] + data[kpts_name][5][1])/2
            if nj in [9, 12]:
                data[kpts_name][8][0] = (data[kpts_name][9][0] + data[kpts_name][12][0])/2
                data[kpts_name][8][1] = (data[kpts_name][9][1] + data[kpts_name][12][1])/2
    elif select[bbox_name] == -1 and select['joints'] != -1:
        select['joints'] = -1