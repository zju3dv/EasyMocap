'''
  @ Date: 2021-03-02 16:12:59
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-02 16:12:59
  @ FilePath: /EasyMocap/scripts/calibration/calib_intri.py
'''
# This script calibrate each intrinsic parameters
from easymocap.mytools import write_intri
import numpy as np
import cv2
import os
from os.path import join
from glob import glob
from easymocap.mytools import read_json, Timer

def calib_intri(path, step):
    camnames = sorted(os.listdir(join(path, 'images')))
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, 'images', cam, '*.jpg')))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        k3ds, k2ds = [], []
        for chessname in chessnames[::step]:
            data = read_json(chessname)
            k3d = np.array(data['keypoints3d'], dtype=np.float32)
            k2d = np.array(data['keypoints2d'], dtype=np.float32)
            if k2d[:, -1].sum() < 0.01:
                continue
            k3ds.append(k3d)
            k2ds.append(np.ascontiguousarray(k2d[:, :-1]))
        gray = cv2.imread(imagenames[0], 0)
        print('>> Detect {}/{:3d} frames'.format(cam, len(k2ds)))
        with Timer('calibrate'):
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                k3ds, k2ds, gray.shape[::-1], None, None)
            cameras[cam] = {
                'K': K,
                'dist': dist # dist: (1, 5)
            }
    write_intri(join(path, 'output', 'intri.yml'), cameras)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    calib_intri(args.path, step=args.step)