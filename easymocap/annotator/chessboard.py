'''
  @ Date: 2021-04-13 16:14:36
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-07-17 16:00:17
  @ FilePath: /EasyMocap/easymocap/annotator/chessboard.py
'''
import numpy as np
import cv2

def getChessboard3d(pattern, gridSize):
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    object_points[:,:2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points[:, [0, 1]] = object_points[:, [1, 0]]
    object_points = object_points * gridSize
    return object_points

colors_chessboard_bar = [
    [0, 0, 255],
    [0, 128, 255],
    [0, 200, 200],
    [0, 255, 0],
    [200, 200, 0],
    [255, 0, 0],
    [255, 0, 250]
]

def get_lines_chessboard(pattern=(9, 6)):
    w, h = pattern[0], pattern[1]
    lines = []
    lines_cols = []
    for i in range(w*h-1):
        lines.append([i, i+1])
        lines_cols.append(colors_chessboard_bar[i//w])
    return lines, lines_cols

def _findChessboardCorners(img, pattern):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern, 
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners

def _findChessboardCornersAdapt(img, pattern):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,21, 2)
    return _findChessboardCorners(img, pattern)

def findChessboardCorners(img, annots, pattern):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCornersAdapt, _findChessboardCorners]:
        ret, corners = func(gray, pattern)
        if ret:break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show