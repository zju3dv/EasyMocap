'''
  @ Date: 2022-07-12 11:55:47
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-14 17:57:48
  @ FilePath: /EasyMocapPublic/easymocap/multistage/mirror.py
'''
import numpy as np
import torch
from ..dataset.mirror import flipPoint2D, flipSMPLPoses, flipSMPLParams
from ..estimator.wrapper_base import bbox_from_keypoints
from .lossbase import Keypoints2D

def calc_vanishpoint(keypoints2d):
    '''
        keypoints2d: (2, N, 3)
    '''
    # weight: (N, 1)
    weight = keypoints2d[:, :, 2:].mean(axis=0)
    conf = weight.mean()
    A = np.hstack([
        keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2],
        -(keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    ])
    b = -keypoints2d[0, :, 0:1]*(keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2]) \
        + keypoints2d[0, :, 1:2] * (keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    b = -b
    A = A * weight
    b = b * weight
    avgInsec = np.linalg.inv(A.T @ A) @ (A.T @ b)
    result = np.zeros(3)
    result[0] = avgInsec[0, 0]
    result[1] = avgInsec[1, 0]
    result[2] = 1
    return result

def calc_mirror_transform(m_):
    """ From mirror vector to mirror matrix
    Args:
        m (bn, 4): (a, b, c, d)
    Returns:
        M: (bn, 3, 4)
    """
    norm = torch.norm(m_[:, :3], dim=1, keepdim=True)
    m = m_[:, :3] / norm
    d = m_[:, 3]
    coeff_mat = torch.zeros((m.shape[0], 3, 4), device=m.device)
    coeff_mat[:, 0, 0] = 1 - 2*m[:, 0]**2
    coeff_mat[:, 0, 1] = -2*m[:, 0]*m[:, 1]
    coeff_mat[:, 0, 2] = -2*m[:, 0]*m[:, 2]
    coeff_mat[:, 0, 3] = -2*m[:, 0]*d
    coeff_mat[:, 1, 0] = -2*m[:, 1]*m[:, 0]
    coeff_mat[:, 1, 1] = 1-2*m[:, 1]**2
    coeff_mat[:, 1, 2] = -2*m[:, 1]*m[:, 2]
    coeff_mat[:, 1, 3] = -2*m[:, 1]*d
    coeff_mat[:, 2, 0] = -2*m[:, 2]*m[:, 0]
    coeff_mat[:, 2, 1] = -2*m[:, 2]*m[:, 1]
    coeff_mat[:, 2, 2] = 1-2*m[:, 2]**2
    coeff_mat[:, 2, 3] = -2*m[:, 2]*d
    return coeff_mat

class InitNormal:
    def __init__(self, static) -> None:
        self.static = static
    
    def __call__(self, body_model, body_params, infos):
        if 'normal' in infos.keys():
            print('>>> Reading normal: {}'.format(infos['normal']))
            return body_params
        kpts = infos['keypoints2d']
        kpts0 = kpts[:, 0]
        kpts1 = flipPoint2D(kpts[:, 1])
        vanish_line = torch.stack([kpts0.reshape(-1, 3), kpts1.reshape(-1, 3)], dim=1)
        MIN_THRES = 0.5
        conf = (vanish_line[:, 0, -1] > MIN_THRES) & (vanish_line[:, 1, -1] > MIN_THRES)
        vanish_line = vanish_line[conf]
        vline0 = vanish_line.numpy().transpose(1, 0, 2)
        vpoint0 = calc_vanishpoint(vline0).reshape(1, 3)
        # 计算点到线的距离进行检查
        # two points line: (x1, y1), (x2, y2) ==> (y-y1)/(x-x1) = (y2-y1)/(x2-x1)
        # A = y2 - y1
        # B = x1 - x2
        # C = x2y1 - x1y2
        # d = abs(ax + by + c)/sqrt(a^2+b^2)
        A_v0 = kpts0[:, :, 1] - vpoint0[0, 1]
        B_v0 = vpoint0[0, 0] - kpts0[:, :, 0]
        C_v0 = kpts0[:, :, 0]*vpoint0[0, 1] - vpoint0[0, 0]*kpts0[:, :, 1]
        distance01 = np.abs(A_v0 * kpts1[:, :, 0] + B_v0 * kpts1[:, :, 1] + C_v0)/np.sqrt(A_v0*A_v0 + B_v0*B_v0)
        A_v1 = kpts1[:, :, 1] - vpoint0[0, 1]
        B_v1 = vpoint0[0, 0] - kpts1[:, :, 0]
        C_v1 = kpts1[:, :, 0]*vpoint0[0, 1] - vpoint0[0, 0]*kpts1[:, :, 1]
        distance10 = np.abs(A_v1 * kpts0[:, :, 0] + B_v1 * kpts0[:, :, 1] + C_v1)/np.sqrt(A_v1*A_v1 + B_v1*B_v1)
        DIST_THRES = 0.05
        for nf in range(kpts.shape[0]):
            # 计算scale
            bbox0 = bbox_from_keypoints(kpts0[nf].cpu().numpy())
            bbox1 = bbox_from_keypoints(kpts1[nf].cpu().numpy())
            bbox_size0 = max(bbox0[2]-bbox0[0], bbox0[3]-bbox0[1])
            bbox_size1 = max(bbox1[2]-bbox1[0], bbox1[3]-bbox1[1])
            valid = (kpts0[nf, :, 2] > 0.3) & (kpts1[nf, :, 2] > 0.3)
            dist01_ = valid*distance01[nf] / bbox_size1
            dist10_ = valid*distance10[nf] / bbox_size0
            # 对于距离异常的点，阈值设定为0.1
            # 抑制掉置信度低的视角的点
            not_valid0 = np.where((dist01_ + dist10_ > DIST_THRES*2) & (kpts0[nf][:, -1] < kpts1[nf][:, -1]))[0]
            not_valid1 = np.where((dist01_ + dist10_ > DIST_THRES*2) & (kpts0[nf][:, -1] > kpts1[nf][:, -1]))[0]
            kpts0[nf, not_valid0] = 0.
            kpts1[nf, not_valid1] = 0.
            if len(not_valid0) > 0:
                print('[mirror] filter {} person 0: {}'.format(nf, not_valid0))
            if len(not_valid1) > 0:
                print('[mirror] filter {} person 1: {}'.format(nf, not_valid1))
        kpts1_ = flipPoint2D(kpts1)
        infos['keypoints2d'] = torch.stack([kpts0, kpts1_], dim=1)
        infos['vanish_point0'] = torch.Tensor(vpoint0)
        K = infos['K'][0]
        normal = np.linalg.inv(K) @ vpoint0.T
        normal = normal.T/np.linalg.norm(normal)
        print('>>> Calculating normal from keypoints: {}'.format(normal[0]))
        infos['normal'] = torch.Tensor(normal)
        mirror = torch.zeros((1, 4))
        # 计算镜子平面到相机的距离
        Th = body_params['Th']
        center = Th.mean(axis=1)
        # 相机原点到两个人中心的连线在normal上的投影
        dist = (center * normal).sum(axis=-1).mean()
        print('>>> Calculating distance from Th: {}'.format(dist))
        mirror[0, 3] = - dist # initial guess
        mirror[:, :3] = infos['normal']
        infos['mirror'] = mirror
        return body_params

class RemoveP1:
    def __init__(self, static) -> None:
        self.static = static
    
    def __call__(self, body_model, body_params, infos):
        for key in body_params.keys():
            if key == 'id': continue
            body_params[key] = body_params[key][:, 0]
        return body_params

class Mirror:
    def __init__(self, key) -> None:
        self.key = key

    def before(self, body_params):
        poses = body_params['poses'][:, 0]
        # append root
        poses = torch.cat([torch.zeros_like(poses[..., :3]), poses], dim=-1)
        poses_mirror = flipSMPLPoses(poses)
        poses = torch.cat([poses[:, None, 3:], poses_mirror[:, None, 3:]], dim=1)
        body_params['poses'] = poses
        return body_params

    def after(self,):
        pass

    def final(self, body_params):
        return self.before(body_params)

class Keypoints2DMirror(Keypoints2D):
    def __init__(self, mirror, opt_normal, **kwargs):
        super().__init__(**kwargs)
        if not mirror.requires_grad:
            self.register_buffer('mirror', mirror)
        else:
            self.mirror = mirror
        self.opt_normal = opt_normal
        k2dall = kwargs['keypoints2d']
        size_all = []
        for nf in range(k2dall.shape[0]):
            for nper in range(2):
                kpts = k2dall[nf, nper]
                bbox = bbox_from_keypoints(kpts.cpu().numpy())
                bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                size_all.append(bbox_size)
        size_all = np.array(size_all).reshape(-1, 2)
        scale = (size_all[:, 0] / size_all[:, 1]).mean()
        print('[loss] mean scale = {} from {} frames, use this to balance the two person'.format(scale, size_all.shape[0]))
        # ATTN: here we use v^2 to suppress the outlier detections
        self.conf = self.conf * self.conf
        self.conf[:, 1] *= scale*scale
    
    def check(self, kpts_est, min_conf=0.3):
        with torch.no_grad():
            M = calc_mirror_transform(self.mirror)
            homo = torch.ones((*kpts_est.shape[:-1], 1), device=kpts_est.device)
            kpts_homo = torch.cat([kpts_est, homo], dim=-1)
            kpts_mirror = flipPoint2D(torch.matmul(M, kpts_homo.transpose(1, 2)).transpose(1, 2))
            kpts = torch.stack([kpts_est, kpts_mirror], dim=1)
            img_points = self.project(kpts)
        conf = (self.conf>min_conf)
        err = self.K[..., 0:1, 0].mean() * torch.norm(img_points - self.keypoints, dim=-1) * conf
        if len(err.shape) == 3:
            err = err.sum(dim=1)
            conf = conf.sum(dim=1)
        err = err.sum(dim=0)/(1e-5 + conf.sum(dim=0))
        return conf, err

    def forward(self, kpts_est, **kwargs):
        if self.opt_normal:
            M = calc_mirror_transform(self.mirror)
        else:
            mirror = torch.cat([self.mirror[:, :3].detach(), self.mirror[:, 3:]], dim=1)
            M = calc_mirror_transform(mirror)
        homo = torch.ones((*kpts_est.shape[:-1], 1), device=kpts_est.device)
        kpts_homo = torch.cat([kpts_est, homo], dim=-1)
        kpts_mirror = flipPoint2D(torch.matmul(M, kpts_homo.transpose(1, 2)).transpose(1, 2))
        kpts = torch.stack([kpts_est, kpts_mirror], dim=1)
        return super().forward(kpts_est=kpts, **kwargs)

class MirrorPoses:
    def __init__(self, ref) -> None:
        self.ref = ref
    
    def __call__(self, body_model, body_params, infos):
        # shapes: (nFrames, 2, nShapes)
        shapes = body_params['shapes'].mean(axis=0).mean(axis=0).reshape(1, 1, -1)
        poses = body_params['poses'][:, 0]
        # append root
        poses = np.concatenate([np.zeros([poses.shape[0], 3]), poses], axis=1)
        poses_mirror = flipSMPLPoses(poses)
        poses = np.concatenate([poses[:, None, 3:], poses_mirror[:, None, 3:]], axis=1)
        body_params['poses'] = poses
        body_params['shapes'] = shapes
        return body_params

class MirrorParams:
    def __init__(self, key) -> None:
        self.key = key

    def start(self, body_params):
        if len(body_params['poses'].shape) == 2:
            return body_params
        for key in body_params.keys():
            if key == 'id': continue
            body_params[key] = body_params[key][:, 0]
        return body_params

    def before(self, body_params):
        return body_params

    def after(self,):
        pass

    def final(self, body_params):
        device = body_params['poses'].device
        body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
        body_params['poses'] = np.hstack((np.zeros_like(body_params['poses'][:, :3]), body_params['poses']))
        params_mirror = flipSMPLParams(body_params, self.infos['mirror'].cpu().numpy())
        params = {}
        for key in params_mirror.keys():
            if key == 'shapes':
                params[key] = body_params[key][:, None]
            else:
                params[key] = np.concatenate([body_params[key][:, None], params_mirror[key][:, None]], axis=-2)
        params['poses'] = params['poses'][..., 3:]
        params = {key:torch.Tensor(val).to(device) for key, val in params.items()}
        return params