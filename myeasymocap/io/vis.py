import os
from typing import Any
import numpy as np
import cv2
from os.path import join
from easymocap.mytools.vis_base import plot_keypoints_auto, merge, plot_bbox, get_rgb, plot_cross
from easymocap.datasets.base import add_logo
from easymocap.mytools.camera_utils import Undistort

def projectPoints(k3d, camera):
    k3d0 = np.ascontiguousarray(k3d[:, :3])
    k3d_rt = np.dot(k3d0, camera['R'].T) + camera['T'].T
    depth = k3d_rt[:, -1:]
    k2d, _ = cv2.projectPoints(k3d0, camera['R'], camera['T'], camera['K'], camera['dist'])
    k2d = np.hstack([k2d[:, 0], k3d[:, -1:]])
    return k2d, depth

class VisBase:
    def __init__(self, scale=1, lw_factor=1, name='vis', mode='none', mode_args={}):
        self.scale = scale
        self.output = '/tmp'
        self.name = name
        self.lw = lw_factor
        self.count = 0
        self.mode = mode
        self.mode_args = mode_args
    
    def merge_and_write(self, vis):
        vis = [v for v in vis if not isinstance(v, str)]
        if self.mode == 'center':
            for i, v in enumerate(vis):
                # crop the center region
                left = int(v.shape[1] - v.shape[0]) // 2
                v = v[:, left:left+v.shape[0], :]
                vis[i] = v
        elif self.mode == 'crop':
            for i, v in enumerate(vis):
                t, b, l, r = self.mode_args[i]
                v = v[t:b, l:r]
                vis[i] = v
        if len(vis) == 0:
            return 0
        if len(vis) == 3: # 只有3个的时候的merge方案：第一个不变，后面两个缩小了放在右边
            vis_0 = vis[0]
            vis_1 = cv2.resize(vis[1], None, fx=0.5, fy=0.5)
            vis_2 = cv2.resize(vis[2], None, fx=0.5, fy=0.5)
            vis_12 = np.vstack([vis_1, vis_2])
            vis = np.hstack([vis_0, vis_12])
        else:
            vis = merge(vis)
        vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale)
        vis = add_logo(vis)
        # TODO: 从输入的Meta里面读入图片名字
        outname = join(self.output, self.name, '{:06d}.jpg'.format(self.count))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, vis)
        self.count += 1

class Vis3D(VisBase):
    def __init__(self, scale, lw_factor=1, name='vis_repro', **kwargs) -> None:
        super().__init__(scale, lw_factor, name, **kwargs)
    
    def __call__(self, images, cameras, keypoints3d=None, results=None):
        # keypoints3d: (nJoints, 4)
        undist = False
        vis_all = []
        for nv in range(len(images)):
            if isinstance(images[nv], str): continue
            camera = {key:cameras[key][nv] for key in ['R', 'T', 'K', 'dist']}
            if undist:
                vis = Undistort.image(images[nv], cameras['K'][nv], cameras['dist'][nv])
                camera['dist'] = np.zeros_like(camera['dist'])
            else:
                vis = images[nv].copy()
            
            if results is None:
                if len(keypoints3d.shape) == 2:
                    keypoints_repro, depth = projectPoints(keypoints3d, {key:cameras[key][nv] for key in ['R', 'T', 'K', 'dist']})
                    plot_keypoints_auto(vis, keypoints_repro, pid=0, use_limb_color=True)
                else:
                    for pid in range(keypoints3d.shape[0]):
                        keypoints_repro, depth = projectPoints(keypoints3d[pid], {key:cameras[key][nv] for key in ['R', 'T', 'K', 'dist']})
                        if (depth < 0.5).all():
                            continue
                        plot_keypoints_auto(vis, keypoints_repro, pid=pid, use_limb_color=True)
            else:
                for res in results:
                    k3d = res['keypoints3d']
                    k3d_rt = np.dot(k3d[:, :3], camera['R'].T) + camera['T'].T
                    keypoints_repro, depth = projectPoints(k3d, camera)
                    depth = k3d_rt[..., -1]
                    if k3d.shape[0] == 1:
                        x, y = keypoints_repro[0,0], keypoints_repro[0,1]
                        plot_cross(vis, x, y, col=get_rgb(res['id']), lw=self.lw, width=self.lw * 5)
                    elif k3d.shape[0] == 2: # limb
                        x1, y1 = keypoints_repro[0,0], keypoints_repro[0,1]
                        x2, y2 = keypoints_repro[1,0], keypoints_repro[1,1]
                        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), get_rgb(res['id']), self.lw)
                    else:
                        plot_keypoints_auto(vis, keypoints_repro, pid=res['id'], use_limb_color=True, lw_factor=self.lw)
                    cv2.putText(vis, '{}'.format(res['id']), (int(keypoints_repro[0,0]), int(keypoints_repro[0,1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, get_rgb(res['id']), self.lw)
            vis_all.append(vis)
        self.merge_and_write(vis_all)

class VisRoot(VisBase):
    def __call__(self, images, pelvis):
        vis = []
        for nv in range(len(images)):
            if isinstance(images[nv], str): continue
            v = images[nv].copy()
            for i in range(pelvis[nv].shape[0]):
                color = get_rgb(i)
                x, y = pelvis[nv][i][0], pelvis[nv][i][1]
                x, y = int(x), int(y)
                plot_cross(v, x, y , col=color, lw=self.lw, width=self.lw * 10)
                cv2.putText(v, '{}'.format(i), (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, self.lw)
            vis.append(v)
        self.merge_and_write(vis)

class VisPAF(VisBase):
    def __call__(self, images, openpose, openpose_paf):
        # openpose [nViews, nJoints, 3]
        # openpose_paf [nViews, dict, MxN]
        vis_limb = [(8, 1)]
        vis = []
        nViews = len(images)
        for nv in range(nViews):
            if isinstance(images[nv], str): continue
            v = images[nv].copy()
            k2d = openpose[nv]
            paf = openpose_paf[nv]
            for (src, dst) in vis_limb:
                # (M, N)
                paf_ = paf[(src, dst)]
                for i in range(paf_.shape[0]):
                    for j in range(paf_.shape[1]):
                        if paf_[i, j] < 0.1:
                            continue
                        x1, y1 = k2d[src][i, :2]
                        x2, y2 = k2d[dst][j, :2]
                        lw = int(paf_[i, j] * 10)
                        cv2.line(v, (int(x1), int(y1)), (int(x2), int(y2)), get_rgb(src), lw)
            vis.append(v)
        self.merge_and_write(vis)
        

class VisBirdEye(VisBase):
    def __init__(self, xranges, yranges, resolution=1024, name='bird', **kwargs):
        super().__init__(name=name, **kwargs)
        self.xranges = xranges
        self.yranges = yranges
        self.resolution = resolution
        self.blank = np.zeros((resolution, resolution, 3), dtype=np.uint8) + 255
        x0, y0 = self.map_x_y(0, 0)
        cv2.line(self.blank, (x0, 0), (x0, resolution), (0, 0, 0), 1)
        cv2.line(self.blank, (0, y0), (resolution, y0), (0, 0, 0), 1)
    
    def map_x_y(self, x, y):
        x = (x - self.xranges[0]) / (self.xranges[1] - self.xranges[0]) * self.resolution
        y = (y - self.yranges[0]) / (self.yranges[1] - self.yranges[0]) * self.resolution
        y = self.resolution - y
        x, y = int(x), int(y)
        return x, y

    def __call__(self, results, cameras):
        vis = self.blank.copy()
        R = cameras['R']
        T = cameras['T']
        # 这里要兼容将来的相机运动的情况，所以不能预先可视化好
        center = - np.einsum('bmn,bnj->bmj', R.swapaxes(1, 2), T)
        for nv in range(center.shape[0]):
            x, y = center[nv, 0], center[nv, 1]
            x, y = self.map_x_y(x, y)
            plot_cross(vis, x, y, col=(0,0,255), lw=self.lw, width=20)
            cv2.putText(vis, 'cam{}'.format(nv), (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), self.lw//4)
        for res in results:
            pid = res['id']
            color = get_rgb(pid)
            x, y, z = res['pelvis'][0, 0], res['pelvis'][0, 1], res['pelvis'][0, 2]
            length = 0.5 * (np.clip(z - 1., 0, 1) + 1)
            length = int(length/(self.xranges[1] - self.xranges[0]) * self.resolution)
            x, y = self.map_x_y(x, y)
            plot_cross(vis, x, y, col=color, lw=self.lw, width=self.lw * 5)
            cv2.rectangle(vis, (x - length, y - length), (x + length, y + length), color, self.lw)
            cv2.putText(vis, '{}'.format(pid), (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, self.lw)
        self.merge_and_write([vis])


class VisMatch(VisBase):
    def __call__(self, images, pelvis, results):
        vis = []
        for nv in range(len(images)):
            if isinstance(images[nv], str): 
                vis.append(images[nv])
                continue
            else:
                vis.append(images[nv].copy())
        for res in results:
            pid = res['id']
            for nv, ind in zip(res['views'], res['indices']):
                v = vis[nv]
                if isinstance(v, str): continue
                x, y = pelvis[nv][ind][0], pelvis[nv][ind][1]
                plot_cross(v, pelvis[nv][ind][0], pelvis[nv][ind][1], col=get_rgb(pid), lw=self.lw, width=self.lw * 5)
                cv2.putText(v, '{}'.format(pid), (int(x), int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, get_rgb(pid), self.lw)
        self.merge_and_write(vis)
    
class Vis_det(VisBase):
    def __call__(self, images, **kwargs):
        vis = []
        for nv in range(len(images)):
            if isinstance(images[nv], str): 
                vis.append(images[nv])
                continue
            else:
                v = images[nv].copy()
                for key, bbox in kwargs.items():
                    _bbox = bbox[nv]
                    for idet in range(_bbox.shape[0]):
                        plot_bbox(v, _bbox[idet], idet)
                vis.append(v)
        self.merge_and_write(vis)

class Vis2D(VisBase):    
    def __call__(self, images, **kwargs):
        if 'keypoints' in kwargs:
            keypoints = kwargs['keypoints']
        else:
            if len(kwargs.keys()) == 1:
                keypoints = list(kwargs.values())[0]
            else:
                raise NotImplementedError
        if 'bbox' in kwargs:
            bbox = kwargs['bbox']
        else:
            bbox = None
        if not isinstance(images, list):
            images = [images]
            keypoints = [keypoints]
            bbox = [bbox]
        vis = []
        for nv in range(len(images)):
            if isinstance(images[nv], str): continue
            k2d = keypoints[nv]
            vis_ = images[nv].copy()
            if len(k2d.shape) == 2:
                plot_keypoints_auto(vis_, k2d, pid=0, use_limb_color=False)
                if bbox is not None:
                    if len(bbox[nv].shape) == 2:
                        plot_bbox(vis_, bbox[nv][0], 0)
                    else:
                        plot_bbox(vis_, bbox[nv], 0)
            else:
                for pid in range(k2d.shape[0]):
                    plot_keypoints_auto(vis_, k2d[pid], pid=pid, use_limb_color=True)
                    if bbox is not None:
                        plot_bbox(vis_, bbox[nv][pid], pid=pid)
            vis.append(vis_)
        self.merge_and_write(vis)