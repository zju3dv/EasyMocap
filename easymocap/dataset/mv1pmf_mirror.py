'''
  @ Date: 2021-01-12 17:12:50
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-14 11:26:36
  @ FilePath: /EasyMocapRelease/easymocap/dataset/mv1pmf_mirror.py
'''
import os
from os.path import join
import numpy as np
import cv2
from .base import ImageFolder
from .mv1pmf import MVBase
from .mirror import calc_mirror_transform, flipSMPLParams, mirrorPoint3D, flipPoint2D, mirror_Rh
from ..mytools.file_utils import get_bbox_from_pose, read_json

class MV1PMF_Mirror(MVBase):
    def __init__(self, root, cams=[], pid=0, out=None, config={}, 
        image_root='images', annot_root='annots', kpts_type='body15',
        undis=True, no_img=False,
        verbose=False) -> None:
        self.mirror = np.array([[0., 1., 0., 0.]])
        super().__init__(root=root, cams=cams, out=out, config=config, 
            image_root=image_root, annot_root=annot_root, 
            kpts_type=kpts_type, undis=undis, no_img=no_img)
        self.pid = pid
        self.verbose = False

    def __str__(self) -> str:
        return 'Dataset for MultiMirror: {} views'.format(len(self.cams))
    
    def write_keypoints3d(self, keypoints3d, nf):
        results = []
        M = self.Mirror[0]
        pid = self.pid
        val = {'id': pid, 'keypoints3d': keypoints3d}
        results.append(val)
        kpts = keypoints3d
        kpts3dm = (M[:3, :3] @ kpts[:, :3].T + M[:3, 3:]).T
        kpts3dm = np.hstack([kpts3dm, kpts[:, 3:]])
        kpts3dm = flipPoint2D(kpts3dm)
        val1 = {'id': pid + 1, 'keypoints3d': kpts3dm}
        results.append(val1)
        super().write_keypoints3d(results, nf)

    def write_smpl(self, params, nf):
        outname = join(self.out, 'smpl', '{:06d}.json'.format(nf))
        results = []
        M = self.Mirror[0]
        pid = self.pid
        val = {'id': pid}
        val.update(params)
        results.append(val)
        # 增加镜子里的人的
        val = {'id': pid + 1}
        val.update(flipSMPLParams(params, self.mirror))
        results.append(val)
        self.writer.write_smpl(results, outname)

    def vis_smpl(self, vertices, faces, images, nf, sub_vis=[], 
        mode='smpl', extra_data=[], add_back=True):
        outname = join(self.out, 'smpl', '{:06d}.jpg'.format(nf))
        render_data = {}
        if len(vertices.shape) == 3:
            vertices = vertices[0]
        pid = self.pid
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        vertices_m = mirrorPoint3D(vertices, self.Mirror[0])
        render_data[pid+1] = {'vertices': vertices_m, 'faces': faces, 
            'vid': pid, 'name': 'human_mirror_{}_{}'.format(nf, pid)}
        
        cameras = {'K': [], 'R':[], 'T':[]}
        if len(sub_vis) == 0:
            sub_vis = self.cams
        for key in cameras.keys():
            cameras[key] = [self.cameras[cam][key] for cam in sub_vis]
        images = [images[self.cams.index(cam)] for cam in sub_vis]
        self.writer.vis_smpl(render_data, images, cameras, outname, add_back=add_back)
    
    def vis_detections(self, images, annots, nf, to_img=True, sub_vis=[]):
        outname = join(self.out, 'detec', '{:06d}.jpg'.format(nf))
        lDetections = []
        nViews = len(images)
        for nv in range(len(images)):
            det = {
                'id': self.pid,
                'bbox': annots['bbox'][nv],
                'keypoints2d': annots['keypoints'][nv]
            }
            det_m = {
                'id': self.pid + 1,
                'bbox': annots['bbox'][nv+nViews],
                'keypoints2d': annots['keypoints'][nv+nViews]
            }
            lDetections.append([det, det_m])
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_keypoints2d_mv(images, lDetections, outname=outname, vis_id=False)

    def vis_repro(self, images, kpts_repro, nf, to_img=True, sub_vis=[]):
        outname = join(self.out, 'repro', '{:06d}.jpg'.format(nf))
        lDetections = []
        for nv in range(len(images)):
            det = {
                'id': -1,
                'keypoints2d': kpts_repro[nv],
                'bbox': get_bbox_from_pose(kpts_repro[nv], images[nv])
            }
            det_mirror = {
                'id': -1,
                'keypoints2d': kpts_repro[nv+len(images)],
                'bbox': get_bbox_from_pose(kpts_repro[nv+len(images)], images[nv])
            }
            lDetections.append([det, det_mirror])
        if len(sub_vis) != 0:
            valid_idx = [self.cams.index(i) for i in sub_vis]
            images = [images[i] for i in valid_idx]
            lDetections = [lDetections[i] for i in valid_idx]
        return self.writer.vis_keypoints2d_mv(images, lDetections, outname=outname, vis_id=False)

    @property
    def Mirror(self):
        M = calc_mirror_transform(self.mirror)
        return M

    @property
    def Pall(self):
        return self.Pall_

    @Pall.setter
    def Pall(self, value):
        M = self.Mirror
        if M.shape[0] == 1 and M.shape[0] != value.shape[0]:
            M = M.repeat(value.shape[0], 0)
        Pall_mirror = np.einsum('bmn,bno->bmo', value, M)
        Pall = np.vstack((value, Pall_mirror))
        self.Pall_ = Pall
        
    def __getitem__(self, index: int):
        images, annots_all = super().__getitem__(index)
        annots0 = self.select_person(annots_all, index, self.pid)
        annots1 = self.select_person(annots_all, index, self.pid + 1)
        # flip points
        # stack it as only one person
        annots = {
            'bbox': np.vstack([annots0['bbox'], annots1['bbox']]),
            'keypoints': np.vstack([annots0['keypoints'], flipPoint2D(annots1['keypoints'])]),
        }
        return images, annots

class ImageFolderMirror(ImageFolder):
    def normal(self, nf):
        annname = join(self.annot_root, self.annotlist[nf])
        data = read_json(annname)
        if 'vanish_point' in data.keys():
            vp1 = np.array(data['vanish_point'][1])
            vp1[2] = 1
            K = self.camera(nf)['K']
            normal = np.linalg.inv(K) @ vp1.reshape(3, 1)
            normal = normal.T / np.linalg.norm(normal)
        else:
            normal = None
        # normal: (1, 3)
        return normal

    def normal_all(self, start, end):
        normals = []
        for nf in range(start, end):
            annname = join(self.annot_root, self.annotlist[nf])
            data = read_json(annname)
            if 'vanish_point' in data.keys():
                vp1 = np.array(data['vanish_point'][1])
                vp1[2] = 1
                K = self.camera(nf)['K']
                normal = np.linalg.inv(K) @ vp1.reshape(3, 1)
                normal = normal.T / np.linalg.norm(normal)
                normals.append(normal)
        # nFrames, 1, 3
        if len(normals) > 0:
            normals = np.stack(normals)
        else:
            normals = None
        return normals

if __name__ == "__main__":
    pass