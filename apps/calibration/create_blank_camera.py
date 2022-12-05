'''
  @ Date: 2021-12-05 15:26:40
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-04 16:47:11
  @ FilePath: /EasyMocapPublic/apps/calibration/create_blank_camera.py
'''
import os
from os.path import join
import numpy as np
from glob import glob
from easymocap.mytools.camera_utils import write_camera
import cv2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--scale', type=int, default=1.2)
    parser.add_argument('--shape', type=int, nargs=2, default=[])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    subs = sorted(os.listdir(join(args.path, 'images')))
    cameras = {}
    for sub in subs:
        if len(args.shape) == 0:
            imgnames = sum([], 
                sorted(glob(join(args.path, 'images', sub, '*.jpg'))) + \
                sorted(glob(join(args.path, 'images', sub, '*.png')))
            )

            imgname = imgnames[0]
            img = cv2.imread(imgname)
            height, width = img.shape[:2]
            print('Read shape {} from image {}'.format(img.shape, imgname))
        else:
            height, width = args.shape
        focal = 1.2*min(height, width) # as colmap
        K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
        cameras[sub] = camera
    write_camera(cameras, args.path)