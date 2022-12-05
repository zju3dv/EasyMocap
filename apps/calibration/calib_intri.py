'''
  @ Date: 2021-03-02 16:12:59
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-11 16:36:12
  @ FilePath: /EasyMocapPublic/apps/calibration/calib_intri.py
'''
# This script calibrate each intrinsic parameters
import shutil
import random
from easymocap.mytools.debug_utils import log, mywarn
from easymocap.mytools.vis_base import plot_points2d
from easymocap.mytools import write_intri, read_json, Timer
import numpy as np
import cv2
import os
from os.path import join
from glob import glob
from easymocap.annotator.chessboard import get_lines_chessboard
from tqdm import tqdm

def read_chess(chessname):
    data = read_json(chessname)
    k3d = np.array(data['keypoints3d'], dtype=np.float32)
    k2d = np.array(data['keypoints2d'], dtype=np.float32)
    if (k2d[:, -1] > 0.).sum() < k2d.shape[0]//2:
        return False, k2d, k3d
    if k2d[:, -1].sum() < k2d.shape[0]:
        valid = k2d[:, -1] > 0.1
        k2d = k2d[valid]
        k3d = k3d[valid]
    # TODO:去除正对相机的
    # TODO:去除各条线不平行的噪声
    return True, k2d, k3d

def pop(k2ds_, k3ds_, valid_idx, imgnames, max_num):
    k2ds = np.stack(k2ds_)
    dist = np.linalg.norm(k2ds[:, None] - k2ds[None, :], axis=-1).mean(axis=-1)
    size = np.linalg.norm(k2ds[:, -1] - k2ds[:, 0], axis=-1)
    dist = dist / size[:, None]
    row = np.arange(dist.shape[0])
    dist[row, row] = 9999.
    col = dist.argmin(axis=0)
    dist_min = dist[row, col]
    indices = dist_min.argsort()[:dist_min.shape[0] - max_num]
    if False:
        img0 = cv2.imread(imgnames[valid_idx[idx]])
        img1 = cv2.imread(imgnames[valid_idx[remove_id]])
        cv2.imshow('01', np.hstack([img0, img1]))
        cv2.waitKey(10)
        print('remove: ', imgnames[valid_idx[remove_id]], imgnames[valid_idx[idx]])
    indices = indices.tolist()
    indices.sort(reverse=True, key=lambda x:col[x])
    removed = set()
    for idx in indices:
        remove_id = col[idx]
        if remove_id in removed:
            continue
        removed.add(remove_id)
        valid_idx.pop(remove_id)
        k2ds_.pop(remove_id)
        k3ds_.pop(remove_id)

def load_chessboards(chessnames, imagenames, max_image, sample_image=-1, out='debug-calib'):
    os.makedirs(out, exist_ok=True)
    k3ds_, k2ds_, imgs = [], [], []
    valid_idx = []
    for i, chessname in enumerate(tqdm(chessnames, desc='read')):
        flag, k2d, k3d = read_chess(chessname)
        if not flag:
            continue
        k3ds_.append(k3d)
        k2ds_.append(k2d)
        valid_idx.append(i)
        if max_image > 0 and len(valid_idx) > max_image + int(max_image * 0.1):
            pop(k2ds_, k3ds_, valid_idx, imagenames, max_num=max_image)
    if sample_image > 0:
        mywarn('[calibration] Load {} images, sample {} images'.format(len(k3ds_), sample_image))
        index = [i for i in range(len(k2ds_))]
        index_sample = random.sample(index, min(sample_image, len(index)))
        valid_idx = [valid_idx[i] for i in index_sample]
        k2ds_ = [k2ds_[i] for i in index_sample]
        k3ds_ = [k3ds_[i] for i in index_sample]
    else:
        log('[calibration] Load {} images'.format(len(k3ds_)))
    for ii, idx in enumerate(valid_idx):
        shutil.copyfile(imagenames[idx], join(out, '{:06d}.jpg'.format(ii)))
    return k3ds_, k2ds_

def calib_intri_share(path, image, ext):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [cam for cam in camnames if os.path.isdir(join(path, image, cam))]

    imagenames = sorted(glob(join(path, image, '*', '*' + ext)))
    chessnames = sorted(glob(join(path, 'chessboard', '*', '*.json')))
    k3ds_, k2ds_ = load_chessboards(chessnames, imagenames, args.num, args.sample, out=join(args.path, 'output'))
    with Timer('calibrate'):
        print('[Info] start calibration with {} detections'.format(len(k2ds_)))
        gray = cv2.imread(imagenames[0], 0)
        k3ds = k3ds_
        k2ds = [np.ascontiguousarray(k2d[:, :-1]) for k2d in k2ds_]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            k3ds, k2ds, gray.shape[::-1], None, None,
            flags=cv2.CALIB_FIX_K3)
        cameras = {}
        for cam in camnames:
            cameras[cam] = {
                'K': K,
                'dist': dist  # dist: (1, 5)
            }
        if True:
            img = cv2.imread(imagenames[0])
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newcameramtx, (w,h), 5)
            for imgname in tqdm(imagenames):
                img = cv2.imread(imgname)
                dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                outname = join(path, 'output', os.path.basename(imgname))
                cv2.imwrite(outname, dst)
        write_intri(join(path, 'output', 'intri.yml'), cameras)

def calib_intri(path, image, ext):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [cam for cam in camnames if os.path.isdir(join(path, image, cam))]
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*'+ext)))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        k3ds_, k2ds_ = load_chessboards(chessnames, imagenames, args.num, out=join(args.path, 'output', cam+'_used'))
        k3ds = k3ds_
        k2ds = [np.ascontiguousarray(k2d[:, :-1]) for k2d in k2ds_]
        gray = cv2.imread(imagenames[0], 0)
        print('>> Camera {}: {:3d} frames'.format(cam, len(k2ds)))
        with Timer('calibrate'):
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                k3ds, k2ds, gray.shape[::-1], None, None,
                flags=cv2.CALIB_FIX_K3)
            cameras[cam] = {
                'K': K,
                'dist': dist  # dist: (1, 5)
            }
    write_intri(join(path, 'output', 'intri.yml'), cameras)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--share_intri', action='store_true')
    parser.add_argument('--remove', action='store_true')
    args = parser.parse_args()
    if args.share_intri:
        calib_intri_share(args.path, args.image, ext=args.ext)
    else:
        calib_intri(args.path, args.image, ext=args.ext)
