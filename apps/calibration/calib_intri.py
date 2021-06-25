'''
  @ Date: 2021-03-02 16:12:59
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-26 23:22:26
  @ FilePath: /EasyMocap/apps/calibration/calib_intri.py
'''
# This script calibrate each intrinsic parameters
from easymocap.mytools.vis_base import plot_points2d
from easymocap.mytools import write_intri, read_json, Timer
import numpy as np
import cv2
import os
from os.path import join
from glob import glob
from easymocap.annotator.chessboard import get_lines_chessboard

def read_chess(chessname):
    data = read_json(chessname)
    k3d = np.array(data['keypoints3d'], dtype=np.float32)
    k2d = np.array(data['keypoints2d'], dtype=np.float32)
    if k2d[:, -1].sum() < 0.01:
        return False, k2d, k3d
    return True, k2d, k3d

def calib_intri_share(path, step):
    camnames = sorted(os.listdir(join(path, 'images')))
    imagenames = sorted(glob(join(path, 'images', '*', '*.jpg')))
    chessnames = sorted(glob(join(path, 'chessboard', '*', '*.json')))
    k3ds_, k2ds_, imgs = [], [], []
    valid_idx = []
    for i, chessname in enumerate(chessnames):
        flag, k2d, k3d = read_chess(chessname)
        k3ds_.append(k3d)
        k2ds_.append(k2d)
        if not flag:
            continue
        valid_idx.append(i)
    MAX_ERROR_PIXEL = 1.
    lines, line_cols = get_lines_chessboard()
    valid_idx = valid_idx[::step]
    len_valid = len(valid_idx)
    cameras = {}
    while True:
        # sample
        imgs = [imagenames[i] for i in valid_idx]
        k3ds = [k3ds_[i] for i in valid_idx]
        k2ds = [np.ascontiguousarray(k2ds_[i][:, :-1]) for i in valid_idx]
        gray = cv2.imread(imgs[0], 0)
        print('>> Detect {:3d} frames'.format(len(valid_idx)))
        with Timer('calibrate'):
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                k3ds, k2ds, gray.shape[::-1], None, None)
        with Timer('check'):
            removed = []
            for i in range(len(imgs)):
                img = cv2.imread(imgs[i])
                points2d_repro, _ = cv2.projectPoints(
                    k3ds[i], rvecs[i], tvecs[i], K, dist)
                points2d_repro = points2d_repro.squeeze()
                points2d = k2ds_[valid_idx[i]]
                err = np.linalg.norm(points2d_repro - points2d[:, :2], axis=1).mean()
                plot_points2d(img, points2d_repro, lines, col=(0, 0, 255), lw=1, putText=False)
                plot_points2d(img, points2d, lines, lw=1, putText=False)
                print(imgs[i], err)
                # cv2.imshow('vis', img)
                # cv2.waitKey(0)
                if err > MAX_ERROR_PIXEL:
                    removed.append(i)
            for i in removed[::-1]:
                valid_idx.pop(i)
        if len_valid == len(valid_idx) or not args.remove:
            print(K)
            print(dist)
            for cam in camnames:
                cameras[cam] = {
                    'K': K,
                    'dist': dist  # dist: (1, 5)
                }
            break
        len_valid = len(valid_idx)
    write_intri(join(path, 'output', 'intri.yml'), cameras)


def calib_intri(path, step):
    camnames = sorted(os.listdir(join(path, 'images')))
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, 'images', cam, '*.jpg')))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        k3ds, k2ds = [], []
        for chessname in chessnames[::step]:
            flag, k2d, k3d = read_chess(chessname)
            if not flag:continue
            k3ds.append(k3d)
            k2ds.append(np.ascontiguousarray(k2d[:, :-1]))
        gray = cv2.imread(imagenames[0], 0)
        print('>> Detect {}/{:3d} frames'.format(cam, len(k2ds)))
        with Timer('calibrate'):
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                k3ds, k2ds, gray.shape[::-1], None, None)
            cameras[cam] = {
                'K': K,
                'dist': dist  # dist: (1, 5)
            }
    write_intri(join(path, 'output', 'intri.yml'), cameras)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--share_intri', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()
    if args.share_intri:
        calib_intri_share(args.path, step=args.step)
    else:
        calib_intri(args.path, step=args.step)
