'''
  @ Date: 2021-03-02 16:13:03
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-27 22:08:18
  @ FilePath: /EasyMocap/scripts/calibration/calib_extri.py
'''
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
from easymocap.mytools import read_intri, write_extri, read_json

def calib_extri(path, intriname):
    assert os.path.exists(intriname), intriname
    intri = read_intri(intriname)
    camnames = list(intri.keys())
    extri = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, 'images', cam, '*.jpg')))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        chessname = chessnames[0]
        data = read_json(chessname)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        k3d[:, 0] *= -1
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        k2d = np.ascontiguousarray(k2d[:, :-1])
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, intri[cam]['K'], intri[cam]['dist'])
        extri[cam] = {}
        extri[cam]['Rvec'] = rvec
        extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
        extri[cam]['T'] = tvec
        center = - extri[cam]['R'].T @ tvec
        print('{} center => {}'.format(cam, center.squeeze()))
    write_extri(join(os.path.dirname(intriname), 'extri.yml'), extri)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--intri', type=str)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    calib_extri(args.path, intriname=args.intri)