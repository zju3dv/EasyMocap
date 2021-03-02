'''
  @ Date: 2021-03-02 16:13:03
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-02 17:06:41
  @ FilePath: /EasyMocap/scripts/calibration/calib_extri.py
'''
import os
from glob import glob
from os.path import join
import cv2
import sys
code_path = join(os.path.dirname(__file__), '..', '..', 'code')
sys.path.append(code_path)
from mytools.camera_utils import read_intri, write_extri, FindChessboardCorners

def calib_extri_pipeline(path, out, resize_rate, debug, args):
    assert os.path.exists(path), path
    intri = read_intri(join(out, 'intri.yml'))
    cameras = [i.split('.')[0] for i in sorted(os.listdir(path))]
    if cameras[0].isdigit():
        cameras.sort(key=lambda x:int(x))
    total = 0
    extri = {}
    for cam in cameras:
        image_names = glob(join(path, '{}.jpg'.format(cam)))
        assert len(image_names) >= 1, '{}/{} has no images'.format(path, cameras)
        infos = FindChessboardCorners([image_names[0]], 
            patternSize=args.pattern, gridSize=args.grid,
            debug=debug, remove=False, resize_rate=resize_rate)
        if len(infos) < 1:
            continue
        info = infos[0]
        ret, rvecs, tvecs = cv2.solvePnP(info['point_object'], info['point_image'], intri[cam]['K'], intri[cam]['dist'])
        extri[cam] = {}
        extri[cam]['Rvec'] = rvecs
        extri[cam]['R'] = cv2.Rodrigues(rvecs)[0]
        extri[cam]['T'] = tvecs
        extri[cam]['center'] = -extri[cam]['R'].T @ tvecs
    write_extri(extri, out, 'extri.yml')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', type=str,dest='path', 
        help='the directory contains the extrinsic images')
    parser.add_argument('-o', '--out', type=str,
        help='output path')
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
        help='The pattern of the chessboard', default=(9, 6))
    parser.add_argument('--grid', type=float, default=0.1, 
        help='The length of the grid size (unit: meter)')
    parser.add_argument('--rate', type=float, default=1, 
        help='scale the original image')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    calib_extri_pipeline(args.path, args.out, args.rate, args.debug, args)