'''
  @ Date: 2021-03-27 19:13:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-15 22:53:23
  @ FilePath: /EasyMocap/apps/calibration/check_calib.py
'''
import cv2
import numpy as np
import os
from os.path import join
from easymocap.mytools import read_json, merge
from easymocap.mytools import read_camera, plot_points2d
from easymocap.mytools import batch_triangulate, projectN3, Undistort
from tqdm import tqdm

POINTS_SQUARE = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [1., 1., 0.],
    [0., 1., 0.]
])

LINES_SQUARE = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
])

def load_cube():
    points3d = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [0., 1., 1.]
    ])
    lines = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ], dtype=np.int)
    points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    return points3d, lines

def merge_points_lines(points3d, lines):
    dist = np.linalg.norm(points3d[:, None, :] - points3d[None, :, :], axis=-1)
    mapid = np.arange(points3d.shape[0])
    for i in range(dist.shape[0]):
        if mapid[i] != i:
            continue
        equal = np.where(dist[i] < 1e-3)[0]
        for j in equal:
            if j == i:
                continue
            mapid[j] = i
    newid = sorted(list(set(mapid)))
    newpoints = points3d[newid]
    for i, newi in enumerate(newid):
        mapid[mapid==newi] = i
    return newpoints, mapid[lines]

def load_grid(xrange=10, yrange=10):
    start = np.array([0., 0., 0.])
    xdir = np.array([1., 0., 0.])
    ydir = np.array([0., 1., 0.])
    stepx = 1.
    stepy = 1.
    points3d, lines = [], []
    for i in range(xrange):
        for j in range(yrange):
            base = start + xdir*i*stepx + ydir*j*stepy
            points3d.append(POINTS_SQUARE+base)
            lines.append(LINES_SQUARE+4*(i*yrange+j))
    points3d = np.vstack(points3d)
    lines = np.vstack(lines)
    return merge_points_lines(points3d, lines)

def load_axes():
    points3d = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., -1.]
    ])
    lines = np.array([
        [0,1],
        [0,2],
        [0,3]
    ], dtype=np.int)
    points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    return points3d, lines

def check_calib(path, out, vis=False, show=False, debug=False):
    if vis:
        out_dir = join(out, 'check')
        os.makedirs(out_dir, exist_ok=True)
    cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
    cameras.pop('basenames')
    total_sum, cnt = 0, 0
    for nf in tqdm(range(10000)):
        imgs = []
        k2ds = []
        for cam, camera in cameras.items():
            if vis:
                imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                img = Undistort.image(img, camera['K'], camera['dist'])
                imgs.append(img)
            annname = join(path, 'chessboard', cam, '{:06d}.json'.format(nf))
            if not os.path.exists(annname):
                break
            data = read_json(annname)
            k2d = np.array(data['keypoints2d'], dtype=np.float32)
            k2d = Undistort.points(k2d, camera['K'], camera['dist'])
            k2ds.append(k2d)
        if len(k2ds) == 0:
            break
        Pall = np.stack([camera['P'] for camera in cameras.values()])
        k2ds = np.stack(k2ds)
        k3d = batch_triangulate(k2ds, Pall)
        kpts_repro = projectN3(k3d, Pall)
        for nv in range(len(k2ds)):
            conf = k2ds[nv][:, -1]
            dist = conf * np.linalg.norm(kpts_repro[nv][:, :2] - k2ds[nv][:, :2], axis=1)
            total_sum += dist.sum()
            cnt += conf.sum()
            if debug:
                print('{:2d}-{:2d}: {:6.2f}/{:2d}'.format(nf, nv, dist.sum(), int(conf.sum())))
            if vis:
                plot_points2d(imgs[nv], kpts_repro[nv], [], col=(0, 0, 255), lw=1, putText=False)
                plot_points2d(imgs[nv], k2ds[nv], [], lw=1, putText=False)
                if show:
                    cv2.imshow('vis', imgs[nv])
                    cv2.waitKey(0)
        if vis:
            imgout = merge(imgs, resize=False)
            outname = join(out, 'check', '{:06d}.jpg'.format(nf))
            cv2.imwrite(outname, imgout)
    print('{:.2f}/{} = {:.2f} pixel'.format(total_sum, int(cnt), total_sum/cnt))

def check_scene(path, out, points3d, lines):
    cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
    cameras.pop('basenames')
    nf = 0
    for cam, camera in cameras.items():
        imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
        assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        img = Undistort.image(img, camera['K'], camera['dist'])
        kpts_repro = projectN3(points3d, camera['P'][None, :, :])[0]
        plot_points2d(img, kpts_repro, lines, col=(0, 0, 255), lw=1, putText=True)
        cv2.imshow('vis', img)
        cv2.waitKey(0)

def check_match(path, out):
    os.makedirs(out, exist_ok=True)
    cameras = read_camera(join(path, 'intri.yml'), join(path, 'extri.yml'))
    cams = cameras.pop('basenames')
    annots = read_json(join(path, 'calib.json'))
    points_global = annots['points_global']
    points3d = np.ones((len(points_global), 4))
    # first triangulate
    points2d = np.zeros((len(cams), len(points_global), 3))
    for i, record in enumerate(points_global):
        for cam, (x, y) in record.items():
            points2d[cams.index(cam), i] = (x, y, 1)
    # 2. undistort
    for nv in range(points2d.shape[0]):
        camera = cameras[cams[nv]]
        points2d[nv] = Undistort.points(points2d[nv], camera['K'], camera['dist'])
    Pall = np.stack([cameras[cam]['P'] for cam in cams])
    points3d = batch_triangulate(points2d, Pall)
    lines = []
    nf = 0
    for cam, camera in cameras.items():
        imgname = join(path, 'images', cam, '{:06d}.jpg'.format(nf))
        assert os.path.exists(imgname), imgname
        img = cv2.imread(imgname)
        img = Undistort.image(img, camera['K'], camera['dist'])
        kpts_repro = projectN3(points3d, camera['P'][None, :, :])[0]
        plot_points2d(img, kpts_repro, lines, col=(0, 0, 255), lw=1, putText=True)
        plot_points2d(img, points2d[cams.index(cam)], lines, col=(0, 255, 0), lw=1, putText=True)
        outname = join(out, cam+'.jpg')
        cv2.imwrite(outname, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, 
        help='the directory contains the extrinsic images')
    parser.add_argument('--out', type=str,
        help='with camera parameters')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cube', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--axes', action='store_true')
    parser.add_argument('--calib', action='store_true')

    args = parser.parse_args()
    if args.cube:
        points, lines = load_cube()
        check_scene(args.path, args.out, points, lines)
    elif args.grid:
        points, lines = load_grid(xrange=15, yrange=14)
        check_scene(args.path, args.out, points, lines)
    elif args.axes:
        points, lines = load_axes()
        check_scene(args.path, args.out, points, lines)
    elif args.calib:
        check_match(args.path, args.out)
    else:
        check_calib(args.path, args.out, args.vis, args.show, args.debug)