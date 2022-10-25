'''
  @ Date: 2022-07-15 19:25:33
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-15 21:00:40
  @ FilePath: /EasyMocapPublic/easymocap/neuralbody/dataset/mirror.py
'''
from os.path import join
import numpy as np
import cv2
from tqdm import trange
import copy
from .mvbase import BaseDataset, read_json, get_bounds
from ...multistage.mirror import calc_mirror_transform
import torch
from .utils_sample import AABBwMask

def mirror_params(params, mirror, T0=np.eye(4, dtype=np.float32)):
    params = params.copy()
    M = np.eye(4, dtype=np.float32)
    M[:3] = calc_mirror_transform(torch.Tensor(mirror))[0].numpy()
    T1 = M @ T0
    T1[0, :] *= -1
    Rh = cv2.Rodrigues(params['Rh'])[0]
    Th = params['Th'].T
    Rnew = T1[:3, :3] @ Rh
    Tnew = T1[:3, :3] @ Th + T1[:3, 3:]
    params['Rh'] = cv2.Rodrigues(Rnew)[0].reshape(1, 3)
    params['Th'] = Tnew.T
    params['vertices'] = params['vertices'] @ T1[:3, :3].T + T1[:3, 3:].T
    return params

class MirrorDataset(BaseDataset):
    def get_allnames(self, root, subs, ranges, image_args):
        assert len(subs) == 1, 'Only support 1 sub-folder'
        infos = []
        index = 0
        nv = 0
        sub = subs[0]
        camera = self.cameras[sub].copy()
        K = camera['K'].copy()
        K[:2] *= image_args.scale
        camera['K'] = K
        # load mirror
        for nnf, nf in enumerate(trange(*ranges, desc='cache parameters')):
            annot = read_json(join(root, 'output-smpl-3d','smpl', sub,  '{:06d}.json'.format(nf)))
            mirror = np.array(annot['mirror'], dtype=np.float32)
            vertices = read_json(join(root, 'output-smpl-3d','vertices', sub,  '{:06d}.json'.format(nf)))
            annots = annot['annots']
            imgname = join(root, image_args.root, sub, '{:06d}{}'.format(nf, image_args.ext))
            info = {
                'imgname': imgname,
                'sub': sub,
                'frame': nf,
                'nf': nnf,
                'nv': nv,
                'index': index,
                'camera': camera,
                'flip_lr': False,
                'pid': 0
            }
            annots[0]['vertices'] = vertices[0]['vertices']
            annots[1]['vertices'] = vertices[0]['vertices']
            info_mirror = info.copy()
            for annot in annots:
                for key, val in annot.items():
                    if key == 'id':continue
                    annot[key] = np.array(val, dtype=np.float32)
            info['params'] = annots[0]
            info_mirror['params'] = mirror_params(annots[0], mirror)
            info_mirror['flip_lr'] = True
            info_mirror['pid'] = 1
            info_mirror['nv'] = 1
            info['bounds'] = get_bounds(info['params']['vertices'], delta=0.1)
            info_mirror['bounds'] = get_bounds(info_mirror['params']['vertices'], delta=0.1)
            info['params']['R'] = cv2.Rodrigues(info['params']['Rh'])[0]
            info_mirror['params']['R'] = cv2.Rodrigues(info_mirror['params']['Rh'])[0]
            infos.append(info)
            infos.append(info_mirror)
        return infos
    
    def get_objects(self, root, info, object_keys, object_args):
        sub, frame, pid = info['sub'], info['frame'], info['pid']
        mskname = join(root, object_args['human_0'].args.reader.mask.root, sub, '{:06d}_{}.png'.format(frame, pid))
        msk = cv2.imread(mskname, 0)
        msk = self.scale_and_undistort(msk, info, undis=False)
        if info['flip_lr']:
            msk = cv2.flip(msk, 1)
        msk = msk > 0
        obj = AABBwMask(split=self.split, bounds=info['bounds'], 
            mask=msk, 
            label=None,
            dilate=False, # 这里假设mask很准，或者使用了patch
            rate_body=0.85)
        for key in ['R', 'Rh', 'Th', 'vertices', 'poses', 'shapes']:
            obj.feature[key] = info['params'][key]
        vertices_canonical = (info['params']['vertices'] - info['params']['Th']) @ info['params']['R'].T.T
        obj.feature['bounds_canonical'] = get_bounds(vertices_canonical, 
            delta=object_args['human_0'].args.reader.vertices.padding)
        return {'human_0': obj}
    
    def read_image(self, imgname, image_args, info, isgray=False, skip_mask=False, mask_global='_0.png'):
        if info['flip_lr']:
            return super().read_image(imgname, image_args, info, isgray, skip_mask, mask_global='_1.png')
        else:
            return super().read_image(imgname, image_args, info, isgray, skip_mask, mask_global)
    
    def augment_rotation(self, info, rot, rotvec=[0., 1., 0.]):
        # method 1: rotate on its self
        rot = rot / 180 * np.pi
        rotvec = np.array([rotvec],dtype=np.float32)
        R = cv2.Rodrigues(rot*rotvec)[0]
        info = copy.deepcopy(info)
        params = info['params']
        T0 = params['Th']
        R0 = cv2.Rodrigues(params['Rh'])[0]
        v0 = params['vertices']
        v1 = (v0 - T0) @ R.T + T0
        params['vertices'] = v1
        R1 = R @ R0
        params['Rh'] = cv2.Rodrigues(R1)[0].reshape(1, 3)
        params['R'] = R1
        info['bounds'] = get_bounds(params['vertices'], delta=0.1)
        return info

class MirrorDatasetDemo(MirrorDataset):
    def __init__(self, keyframes, **cfg):
        self.keyframes = keyframes
        # 注意：初始化里面调用了get_allnames，所以要预先写进去
        super().__init__(**cfg)
    
    def __len__(self):
        return super().__len__()//2
    
    def get_allnames(self, root, subs, ranges, image_args):
        infos = super().get_allnames(root, subs, ranges, image_args)
        infos_new = []
        for nf in range(len(infos)//2):
            infos_l = infos[2*nf]
            infos_r = infos[2*nf+1]
            if nf in self.keyframes:
                for rot in range(0, 360, 4):
                    infos_r_ = self.augment_rotation(infos_r, rot, rotvec=[0,1,0])
                    infos_new.append(infos_l)
                    infos_new.append(infos_r_)
            else:
                infos_new.append(infos_l)
                infos_new.append(infos_r)
        return infos_new
    
    def __getitem__(self, index):
        left = super().__getitem__(2*index)
        right = super().__getitem__(2*index+1)
        return {'left': left, 'right': right, 'meta': {'type': 'mirror'}}

class MirrorDatasetDemoCool(BaseDataset):
    def __init__(self, **cfg):
        super().__init__(**cfg)

    def get_allnames(self, root, subs, ranges, image_args):
        assert len(subs) == 1, 'Only support 1 sub-folder'
        infos = []
        index = 0
        nv = 0
        sub = subs[0]
        camera = self.cameras[sub].copy()
        K = camera['K'].copy()
        K[:2] *= image_args.scale
        camera['K'] = K
        # load mirror
        for nnf, nf in enumerate(trange(*ranges, desc='cache parameters')):
            annot = read_json(join(root, 'output-smpl-3d','smpl', sub,  '{:06d}.json'.format(nf)))
            mirror = np.array(annot['mirror'], dtype=np.float32)
            vertices = read_json(join(root, 'output-smpl-3d','vertices', sub,  '{:06d}.json'.format(nf)))
            # only use outer person
            annots = annot['annots'][:1]
            imgname = join(root, image_args.root, sub, '{:06d}{}'.format(nf, image_args.ext))
            info = {
                'imgname': imgname,
                'sub': sub,
                'frame': nf,
                'nf': nnf,
                'nv': nv,
                'index': index,
                'camera': camera,
                'flip_lr': False,
                'pid': 0
            }
            annots[0]['vertices'] = vertices[0]['vertices']
            for annot in annots:
                for key, val in annot.items():
                    if key == 'id':continue
                    annot[key] = np.array(val, dtype=np.float32)
            info['params'] = annots[0]
            # TODO: augment the rotation
            # method 1: rotate on its self
            rot = nnf / 180 * np.pi
            rotvec = np.array([[0., 1., 0.]],dtype=np.float32)
            R = cv2.Rodrigues(rot*rotvec)[0]
            T0 = info['params']['Th']
            R0 = cv2.Rodrigues(info['params']['Rh'])[0]
            v0 = info['params']['vertices']
            v1 = (v0 - T0) @ R.T + T0
            info['params']['vertices'] = v1
            R1 = R @ R0
            info['params']['Rh'] = cv2.Rodrigues(R1)[0].reshape(1, 3)
            info['params']['R'] = R1
            info['bounds'] = get_bounds(info['params']['vertices'], delta=0.1)
            infos.append(info)
        return infos

    def get_objects(self, root, info, object_keys, object_args):
        sub, frame, pid = info['sub'], info['frame'], info['pid']
        mskname = join(root, object_args['human_0'].args.reader.mask.root, sub, '{:06d}_{}.png'.format(frame, pid))
        msk = cv2.imread(mskname, 0)
        msk = self.scale_and_undistort(msk, info, undis=False)
        if info['flip_lr']:
            msk = cv2.flip(msk, 1)
        msk = msk > 0
        obj = AABBwMask(split=self.split, bounds=info['bounds'], 
            mask=msk, 
            label=None,
            dilate=True,
            rate_body=0.85)
        for key in ['R', 'Rh', 'Th', 'vertices', 'poses', 'shapes']:
            obj.feature[key] = info['params'][key]
        return {'human_0': obj}