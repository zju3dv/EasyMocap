'''
  @ Date: 2021-03-27 19:13:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-11 16:47:10
  @ FilePath: /EasyMocapPublic/apps/calibration/check_calib.py
'''
from easymocap.mytools.debug_utils import myerror, mywarn
from easymocap.mytools.file_utils import myarray2string
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

def load_cube(grid_size=1, **kwargs):
    min_x, min_y, min_z = (0, 0, 0.)
    max_x, max_y, max_z = (grid_size, grid_size, grid_size)
    # min_x, min_y, min_z = (-0.75, -0.9, 0.)
    # max_x, max_y, max_z = (0.75, 0.7, 0.9)
    # # 灯光球场篮球:
    # min_x, min_y, min_z = (-7.5, -2.89, 0.)
    # max_x, max_y, max_z = (7.5, 11.11, 2.)
    # # 4d association:
    # min_x, min_y, min_z = (-1.6, -1.6, 0.)
    # max_x, max_y, max_z = (1.5, 1.6, 2.4)
    # min_x, min_y, min_z = (-2.45, -4., 0.)
    # max_x, max_y, max_z = (1.65, 2.45, 2.6)
    
    points3d = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
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
    ], dtype=np.int64)
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

def load_grid(xrange=28, yrange=15, step=1, two=False, **kwargs):
    start = np.array([0., 0., 0.])
    xdir = np.array([1., 0., 0.])
    ydir = np.array([0., 1., 0.])
    stepx = step
    stepy = step
    points3d, lines = [], []
    if two:
        start_x = -xrange
        start_y = -yrange
    else:
        start_x = 0
        start_y = 0
    for i in range(start_x, xrange):
        for j in range(start_y, yrange):
            base = start + xdir*i*stepx + ydir*j*stepy
            points3d.append(POINTS_SQUARE+base)
            lines.append(LINES_SQUARE+4*((i-start_x)*(yrange-start_y)+(j-start_y)))
    points3d = np.vstack(points3d)
    lines = np.vstack(lines)
    return merge_points_lines(points3d, lines)

def load_human(path, pid, nf=0, camnames=[], annot='annots'):
    points = []
    nvs = []
    annot_ = annot
    for nv, sub in enumerate(camnames):
        annotname = join(path, annot_, sub, '{:06d}.json'.format(nf))
        if not os.path.exists(annotname):
            print('[Warn] Not exist ', annotname)
            continue
        annots = read_json(annotname)
        if isinstance(annots, dict):
            annots = annots['annots']
        annot = [d for d in annots if d['personID'] == pid]
        if len(annot) == 0:
            continue
        pts = np.array(annot[0]['keypoints'])
        if args.hand:
            handl = np.array(annot[0]['handl2d'])
            handr = np.array(annot[0]['handr2d'])
            pts = np.vstack([pts, handl, handr])
        points.append(pts)
        nvs.append(nv)
    points = np.stack(points)
    results = np.zeros((len(camnames), *points.shape[1:]))
    results[nvs] = points
    from easymocap.dataset.config import CONFIG
    lines = CONFIG['body25']['kintree']
    return results, lines

class BaseCheck:
    def __init__(self, path, out, mode='cube', ext='.jpg', sub=[]) -> None:
        cameras = read_camera(join(out, 'intri.yml'), join(out, 'extri.yml'))
        cameras.pop('basenames')
        self.outdir = join(out, mode)
        self.cameras = cameras
        if len(sub) == 0:
            self.camnames = sorted(list(cameras.keys()))
        else:
            self.camnames = sub
        if args.prefix is not None:
            for c in self.camnames:
                self.cameras[c.replace(args.prefix, '')] = self.cameras.pop(c)
            self.camnames = [c.replace(args.prefix, '') for c in self.camnames]
        print('[check] cameras: ', self.camnames)
        zaxis = np.array([0., 0., 1.]).reshape(3, 1)
        for cam in self.camnames:
            camera = cameras[cam]
            center = -camera['R'].T @ camera['T']
            # 
            lookat = camera['R'].T @ (zaxis - camera['T'])
            print(' - {}: center = {}, look at = {}'.format(cam, np.round(center.T, 3), np.round(lookat.T, 3)))

        self.path = path
        self.kpts2d = None
        self.ext = ext
        self.errors = []
    
    def check(self, points3d, lines, nf = 0, show=False, write=True):
        if write:
            os.makedirs(self.outdir, exist_ok=True)
        conf3d = points3d[:, -1]
        p3d = np.ascontiguousarray(points3d[:, :3])
        errors = []
        for nv, cam in enumerate(self.camnames):
            camera = self.cameras[cam]
            if show or write:
                imgname = join(self.path, 'images', cam, '{:06d}{}'.format(nf, self.ext))
                if not os.path.exists(imgname):
                    imgname = join(self.path, 'images', cam, '{:08d}{}'.format(nf, self.ext))
                    if not os.path.exists(imgname):
                        print('[WARN] Not exist', imgname)
                        continue
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                img = Undistort.image(img, camera['K'], camera['dist'])
            if False:
                points2d_repro, xxx = cv2.projectPoints(p3d, cv2.Rodrigues(camera['R'])[0], camera['T'], camera['K'], camera['dist'])
                kpts_repro = points2d_repro.squeeze()
            else:
                kpts_repro = projectN3(p3d, [camera['P']])[0]
            if self.kpts2d is not None:
                k2d = self.kpts2d[nv]
                k2d = Undistort.points(k2d, camera['K'], camera['dist'])
                valid = (conf3d > 0.)&(k2d[:, 2] > 0.)
                # print(kpts_repro)
                # import ipdb; ipdb.set_trace()
                if k2d[:, 2].sum() > 0.:
                    diff = np.linalg.norm(k2d[:, :2] - kpts_repro[:, :2], axis=1) * valid
                    print('[Check] {}: {} points, {:3.2f} pixels， max is {}, {:3.2f} pixels'.format(cam, valid.sum(), diff.sum()/valid.sum(), diff.argmax(), diff.max()))
                    diff = diff.sum()/valid.sum()
                    errors.append(diff)
                    self.errors.append((diff, nv, nf))
                    if show or write:
                        plot_points2d(img, k2d, lines, col=(0, 255, 0), lw=1, putText=False)
            else:
                k2d = np.zeros((10, 3))
            if show or write:
                if points3d.shape[-1] == 4:
                    conf = points3d[..., -1:] > 0.01
                elif points3d.shape[-1] == 3:
                    conf = np.ones_like(points3d[..., -1:])
                kpts_vis = np.hstack((kpts_repro[:, :2], conf))
                # for i in range(kpts_vis.shape[0]):
                #     print('{}: {}, {}, {}'.format(i, *kpts_vis[i]))
                plot_points2d(img, kpts_vis, lines, col=(0, 0, 255), lw=1, putText=args.text, style='+')
                for i in range(kpts_vis.shape[0]):
                    if k2d[i][-1] < 0.1:continue
                    cv2.line(img, (int(kpts_vis[i][0]), int(kpts_vis[i][1])), (int(k2d[i][0]), int(k2d[i][1])), (0,0,0), thickness=2)
            not_skip_unvis = True
            if show and (k2d[:, 2].sum()>0 or not_skip_unvis):
                vis = img
                if vis.shape[0] > 1000:
                    vis = cv2.resize(vis, None, fx=1000/vis.shape[0], fy=1000/vis.shape[0])
                cv2.imshow('vis', vis)
                cv2.waitKey(0)
            if write:
                outname = join(self.outdir, '{}_{:06d}.jpg'.format(cam, nf))
                cv2.imwrite(outname, img)
        if len(errors) > 0:
            print('[Check] Mean error: {:3.2f} pixels'.format(sum(errors)/len(errors)))

    def summary(self):
        errors = self.errors
        if len(errors) > 0:
            errors.sort(key=lambda x:-x[0])
            print('[Check] Total {} frames Mean error: {:3.2f} pixels, max: {:3.2f} in cam "{}" frame {}'.format(len(errors), sum([e[0] for e in errors])/len(errors), errors[0][0], self.camnames[errors[0][1]], self.errors[0][2]))

class QuanCheck(BaseCheck):
    def __init__(self, path, out, mode, ext, sub=[]) -> None:
        super().__init__(path, out, mode, ext, sub)
    
    def triangulate(self, k2ds, gt=None):
        # k2ds: (nViews, nPoints, 3)
        self.kpts2d = k2ds
        k2dus = []
        for nv in range(k2ds.shape[0]):
            camera = self.cameras[self.camnames[nv]]
            k2d = k2ds[nv].copy()
            k2du = Undistort.points(k2d, camera['K'], camera['dist'])
            k2dus.append(k2du)
        Pall = np.stack([self.cameras[cam]['P'] for cam in self.camnames])
        k2dus = np.stack(k2dus)
        k3d = batch_triangulate(k2dus, Pall)
        if gt is not None:
            if gt.shape[0] < k3d.shape[0]: # gt少了点
                gt = np.vstack([gt, np.zeros((k3d.shape[0]-gt.shape[0], 3))])
            valid = np.where(k3d[:, -1] > 0.)[0]
            err3d = np.linalg.norm(k3d[valid, :3] - gt[valid], axis=1)
            print('[Check3D] mean error: {:.2f}mm'.format(err3d.mean()*1000))
        return k3d

def load2d_ground(path, nf=0, camnames=[]):
    k2ds = []
    k3d = None
    MAX_POINTS = 0
    for cam in sorted(camnames):
        annname = join(path, cam, '{:06d}.json'.format(nf))
        if not os.path.exists(annname):
            mywarn(annname + ' not exists')
        data = read_json(annname)
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        if k2d.shape[0] > MAX_POINTS:
            MAX_POINTS = k2d.shape[0]
        k2ds.append(k2d)
    for i, k2d in enumerate(k2ds):
        if k2d.shape[0] < MAX_POINTS:
            k2ds[i] = np.vstack([k2d, np.zeros((MAX_POINTS-k2d.shape[0], 3))])
    k2ds = np.stack(k2ds)
    conf = k2ds[:, :, 2].sum(axis=1)
    if (conf>0).sum() < 2:
        return False, None, None
    return True, k2ds, k3d

def read_match2d_file(file, camnames):
    points = read_json(file)['points_global']
    match2d = np.zeros((len(camnames), len(points), 3))
    for npo in range(match2d.shape[1]):
        for key, (x, y) in points[npo].items():
            if key not in camnames:
                continue
            match2d[camnames.index(key), npo] = [x, y, 1.]
    return True, match2d, np.zeros((match2d.shape[1], 3))

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
                for ext in ['jpg', 'png']:
                    imgname = join(path, 'images', cam, '{:06d}.{}'.format(nf, ext))
                    if not os.path.exists(imgname):
                        continue
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
                kpts_repro_vis = np.hstack((kpts_repro[nv][:, :2], conf[:, None]))
                plot_points2d(imgs[nv], kpts_repro_vis, [], col=(0, 0, 255), lw=1, putText=False)
                plot_points2d(imgs[nv], k2ds[nv], [], lw=1, putText=False)
                for i in range(kpts_repro_vis.shape[0]):
                    cv2.line(imgs[nv], kpts_repro_vis[i], k2ds[nv][i], (0,0,0), thickness=1)
                if show:
                    cv2.imshow('vis', imgs[nv])
                    cv2.waitKey(0)
        if vis:
            imgout = merge(imgs, resize=False)
            outname = join(out, 'check', '{:06d}.jpg'.format(nf))
            cv2.imwrite(outname, imgout)
    print('{:.2f}/{} = {:.2f} pixel'.format(total_sum, int(cnt), total_sum/cnt))

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
        for i in range(kpts_repro_vis.shape[0]):
            cv2.line(imgs[nv], kpts_repro[i], points2d[cams.index(cam)][i], (0,0,0), thickness=1)
        outname = join(out, cam+'.jpg')
        cv2.imwrite(outname, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, 
        help='the directory contains the extrinsic images')
    parser.add_argument('--sub', type=str,
        default=[], nargs='+')
    parser.add_argument('--out', type=str,
        help='with camera parameters')
    parser.add_argument('--mode', type=str, default='cube',
        help='with camera parameters')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--grid_x', type=int, default=3)
    parser.add_argument('--grid_y', type=int, default=3)
    parser.add_argument('--grid_step', type=float, default=1.)
    parser.add_argument('--grid_two', action='store_true')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--hand', action='store_true')
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--annot', type=str, default='annots')
    parser.add_argument('--calib', action='store_true')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--print3d', action='store_true')
    parser.add_argument('--gt', action='store_true')
    
    args = parser.parse_args()
    if args.mode in ['cube', 'grid']:
        points, lines = {'cube': load_cube, 'grid': load_grid}[args.mode](
            xrange=args.grid_x, yrange=args.grid_y, step=args.grid_step, two=args.grid_two,
            grid_size=args.grid_step
        )
        print('Check {} points'.format(points.shape))
        checker = BaseCheck(args.path, args.out, args.mode, args.ext)
        checker.check(points, lines, args.frame, show=args.show, write=args.write)
    elif args.mode in ['gcp', 'match']:
        checker = QuanCheck(args.path, args.out, args.mode, args.ext)
        lines = []
        if args.mode == 'match':
            for nf in range(0, 10000, args.step):
                # try:
                flag, k2ds, gt3d = load2d_ground(join(args.path, args.annot), nf=nf, camnames=checker.camnames)
                # except:
                    # myerror('{} not exist'.format(join(args.path, args.annot, '{:06d}.json'.format(nf))))
                    # break
                if not flag:continue
                points = checker.triangulate(k2ds, gt=gt3d)
                if args.print3d:
                    valid = points[:, -1] > 0.01
                    points_ = points[valid]
                    np.savetxt(join(args.out, 'points3d.txt'), points_, fmt='%10.5f')
                    print(myarray2string(points_, indent=0))
                    norm = np.linalg.norm(points_, axis=1)
                    print('[calib] max norm={}, min norm={}'.format(norm.max(), norm.min()))
                checker.check(gt3d if args.gt else points, lines, nf, show=args.show, write=args.write)
            checker.summary()
        elif args.mode == 'gcp':
            flag, k2ds, gt3d = read_match2d_file(join(args.path, 'calib.json'), camnames=checker.camnames)
            points = checker.triangulate(k2ds, gt=gt3d)
            print(myarray2string(points, indent=4))
            checker.check(gt3d if args.gt else points, lines, 0, show=args.show, write=args.write)
        else:
            flag, k2ds, gt3d = load2d_ground(join(args.path, 'chessboard'), camnames=checker.camnames)
            points = checker.triangulate(k2ds, gt=gt3d)
            checker.check(gt3d if args.gt else points, lines, 0, show=args.show, write=args.write)
    elif args.mode == 'human':
        checker = QuanCheck(args.path, args.out, args.mode, args.ext, sub=args.sub)
        points, lines = load_human(args.path, pid=args.pid, nf=args.frame, camnames=checker.camnames, annot=args.annot)
        points = checker.triangulate(points, gt=None)
        print('[calib] check human')
        print(myarray2string(points, indent=0))
        print('[calib]limblength: {}'.format(np.linalg.norm(points[1, :3] - points[8, :3])))
        checker.check(points, lines, args.frame, show=args.show, write=args.write)
    elif args.calib:
        check_match(args.path, args.out)
    else:
        check_calib(args.path, args.out, args.vis, args.show, args.debug)