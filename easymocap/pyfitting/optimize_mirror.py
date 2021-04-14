'''
  @ Date: 2021-03-05 15:21:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-31 23:02:58
  @ FilePath: /EasyMocap/easymocap/pyfitting/optimize_mirror.py
'''
from .optimize_simple import _optimizeSMPL, deepcopy_tensor, get_prepare_smplx, dict_of_tensor_to_numpy
from .lossfactory import LossRepro, LossInit, LossSmoothBody, LossSmoothPoses, LossSmoothBodyMulti, LossSmoothPosesMulti
from ..dataset.mirror import flipSMPLPoses, flipPoint2D, flipSMPLParams
import torch
import numpy as np
    # 这里存在几种技术方案:
    #   1. theta, beta, R, T, (a, b, c, d)          || L_r
    #   2. theta, beta, R, T, R', T'                || L_r, L_s
    #   3. theta, beta, R, T, theta', beta', R', T' || L_r, L_s

def flipSMPLPosesV(params, reverse=False):
    # 前面部分是外面的人，后面部分是镜子里的人
    nFrames = params['poses'].shape[0] // 2
    if reverse:
        params['poses'][:nFrames] = flipSMPLPoses(params['poses'][nFrames:])
    else:
        params['poses'][nFrames:] = flipSMPLPoses(params['poses'][:nFrames])
    return params

def flipSMPLParamsV(params, mirror):
    params_mirror = flipSMPLParams(params, mirror)
    params_new = {}
    for key in params.keys():
        if key == 'shapes':
            params_new['shapes'] = params['shapes']
        else:
            params_new[key] = np.vstack([params[key], params_mirror[key]])
    return params_new

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

class LossKeypointsMirror2D(LossRepro):
    def __init__(self, keypoints2d, bboxes, Pall, cfg) -> None:
        super().__init__(bboxes, keypoints2d, cfg)
        self.Pall = torch.Tensor(Pall).to(cfg.device)
        self.nJoints = keypoints2d.shape[-2]
        self.nViews, self.nFrames = self.keypoints2d.shape[0], self.keypoints2d.shape[1]
        self.kpt_homo = torch.ones((keypoints2d.shape[0]*keypoints2d.shape[1], keypoints2d.shape[2], 1), device=cfg.device)
        self.norm = 'l2'

    def residual(self, kpts_est):
        # kpts_est: (2xnFrames, nJoints, 3)
        kpts_homo = torch.cat([kpts_est[..., :self.nJoints, :], self.kpt_homo], dim=2)
        point_cam = torch.einsum('ab,fnb->fna', self.Pall, kpts_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        img_points = img_points.view(self.nViews, self.nFrames, self.nJoints, 2)
        residual = (img_points - self.keypoints2d) * self.conf
        return residual

    def __call__(self, kpts_est, **kwargs):
        "reprojection error for mirror"
        # kpts_est: (2xnFrames, 25, 3)
        kpts_homo = torch.cat([kpts_est[..., :self.nJoints, :], self.kpt_homo], dim=2)
        point_cam = torch.einsum('ab,fnb->fna', self.Pall, kpts_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        img_points = img_points.view(self.nViews, self.nFrames, self.nJoints, 2)
        return super().__call__(img_points)/self.nViews/self.nFrames

    def __str__(self) -> str:
        return 'Loss function for Reprojection error of Mirror'

class LossKeypointsMirror2DDirect(LossKeypointsMirror2D):
    def __init__(self, keypoints2d, bboxes, Pall, normal=None, cfg=None, mirror=None) -> None:
        super().__init__(keypoints2d, bboxes, Pall, cfg)
        nFrames = 1
        if mirror is None:
            self.mirror = torch.zeros([nFrames, 4], device=cfg.device)
            if normal is not None:
                self.mirror[:, :3] = torch.Tensor(normal).to(cfg.device)
            else:
                # roughly initialize the mirror => n = (0, -1, 0)
                self.mirror[:, 2] = 1.
            self.mirror[:, 3] = -10.
        else:
            self.mirror = torch.Tensor(mirror).to(cfg.device)
        self.norm = 'l2'

    def __call__(self, kpts_est, **kwargs):
        "reprojection error for direct mirror ="
        # kpts_est: (nFrames, 25, 3)
        M = calc_mirror_transform(self.mirror)
        if M.shape[0] != kpts_est.shape[0]:
            M = M.expand(kpts_est.shape[0], -1, -1)
        homo = torch.ones((kpts_est.shape[0], kpts_est.shape[1], 1), device=kpts_est.device)
        kpts_homo = torch.cat([kpts_est, homo], dim=2)
        kpts_mirror = flipPoint2D(torch.bmm(M, kpts_homo.transpose(1, 2)).transpose(1, 2))
        # 视频的时候注意拼接的顺序
        kpts_new = torch.cat([kpts_est, kpts_mirror])
        # 使用镜像进行翻转
        return super().__call__(kpts_new)

    def __str__(self) -> str:
        return 'Loss function for Reprojection error of Mirror '

class LossMirrorSymmetry:
    def __init__(self, N_JOINTS=25, normal=None, cfg=None) -> None:
        idx0, idx1 = np.meshgrid(np.arange(N_JOINTS), np.arange(N_JOINTS))
        idx0, idx1 = idx0.reshape(-1), idx1.reshape(-1)
        idx_diff = np.where(idx0!=idx1)[0]
        self.idx00, self.idx11 = idx0[idx_diff], idx1[idx_diff]
        self.N_JOINTS = N_JOINTS
        self.idx0 = idx0
        self.idx1 = idx1
        if normal is not None:
            self.normal = torch.Tensor(normal).to(cfg.device)
            self.normal = self.normal.expand(-1, N_JOINTS, -1)
        else:
            self.normal = None
        self.device = cfg.device
    
    def parallel_mirror(self, kpts_est, **kwargs):
        "encourage parallel to mirror"
        # kpts_est: (nFramesxnViews, nJoints, 3)
        if self.normal is None:
            return torch.tensor(0.).to(self.device)
        nFrames = kpts_est.shape[0] // 2
        kpts_out = kpts_est[:nFrames, ...]
        kpts_in = kpts_est[nFrames:, ...]
        kpts_in = flipPoint2D(kpts_in)
        direct = kpts_in - kpts_out
        direct_norm = direct/torch.norm(direct, dim=-1, keepdim=True)
        loss = torch.sum(torch.norm(torch.cross(self.normal, direct_norm), dim=2))
        return loss / nFrames / kpts_est.shape[1]

    def parallel_self(self, kpts_est, **kwargs):
        "encourage parallel to self"
        # kpts_est: (nFramesxnViews, nJoints, 3)
        nFrames = kpts_est.shape[0] // 2
        kpts_out = kpts_est[:nFrames, ...]
        kpts_in = kpts_est[nFrames:, ...]
        kpts_in = flipPoint2D(kpts_in)
        direct = kpts_in - kpts_out
        direct_norm = direct/torch.norm(direct, dim=-1, keepdim=True)
        loss = torch.sum(torch.norm(
            torch.cross(direct_norm[:, self.idx0, :], direct_norm[:, self.idx1, :]), dim=2))/self.idx0.shape[0]
        return loss / nFrames
    
    def vertical_self(self, kpts_est, **kwargs):
        "encourage vertical to self"
        # kpts_est: (nFramesxnViews, nJoints, 3)
        nFrames = kpts_est.shape[0] // 2
        kpts_out = kpts_est[:nFrames, ...]
        kpts_in = kpts_est[nFrames:, ...]
        kpts_in = flipPoint2D(kpts_in)
        direct = kpts_in - kpts_out
        direct_norm = direct/torch.norm(direct, dim=-1, keepdim=True)
        mid_point = (kpts_in + kpts_out)/2
        
        inner = torch.abs(torch.sum((mid_point[:, self.idx00, :] - mid_point[:, self.idx11, :])*direct_norm[:, self.idx11, :], dim=2))
        loss = torch.sum(inner)/self.idx00.shape[0]
        return loss / nFrames

    def __str__(self) -> str:
        return 'Loss function for Mirror Symmetry'

class MirrorLoss():
    def __init__(self, N_JOINTS=25) -> None:
        N_JOINTS = min(N_JOINTS, 25)
        idx0, idx1 = np.meshgrid(np.arange(N_JOINTS), np.arange(N_JOINTS))
        idx0, idx1 = idx0.reshape(-1), idx1.reshape(-1)
        idx_diff = np.where(idx0!=idx1)[0]
        self.idx00, self.idx11 = idx0[idx_diff], idx1[idx_diff]
        self.N_JOINTS = N_JOINTS
        self.idx0 = idx0
        self.idx1 = idx1

    def loss(self, lKeypoints, weight_loss):
        loss_dict = {}
        for key in ['parallel_self', 'parallel_mirror', 'vertical_self']:
            if weight_loss[key] > 0.:
                loss_dict[key] = 0.
        # mirror loss for two person
        kpts0 = lKeypoints[0][..., :self.N_JOINTS, :]
        kpts1 = flipPoint(lKeypoints[1][..., :self.N_JOINTS, :])
        # direct: (N, 25, 3)
        direct = kpts1 - kpts0
        direct_norm = direct/torch.norm(direct, dim=2, keepdim=True)
        if weight_loss['parallel_self'] > 0.:
            loss_dict['parallel_self'] += torch.sum(torch.norm(
                torch.cross(direct_norm[:, self.idx0, :], direct_norm[:, self.idx1, :]), dim=2))/self.idx0.shape[0]
        mid_point = (kpts0 + kpts1)/2
        if weight_loss['vertical_self'] > 0:
            inner = torch.abs(torch.sum((mid_point[:, self.idx00, :] - mid_point[:, self.idx11, :])*direct_norm[:, self.idx11, :], dim=2))
            loss_dict['vertical_self'] += torch.sum(inner)/self.idx00.shape[0]
        return loss_dict

def optimizeMirrorDirect(body_model, params, bboxes, keypoints2d, Pall, normal, weight, cfg):
    """ 
        simple function for optimizing mirror
        # 先写图片的
    Args:
        body_model (SMPL model)
        params (DictParam): poses(2, 72), shapes(1, 10), Rh(2, 3), Th(2, 3)
        bboxes (nFrames, nViews, nJoints, 4): 2D bbox of each view，输入的时候是按照时序叠起来的
        keypoints2d (nFrames, nViews, nJoints, 4): 2D keypoints of each view，输入的时候是按照时序叠起来的
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    nViews, nFrames = keypoints2d.shape[:2]
    assert nViews == 2, 'Please make sure that there exists only 2 views'
    # keep the parameters of the real person
    for key in ['poses', 'Rh', 'Th']:
        # select the parameters of first person
        params[key] = params[key][:nFrames]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
    ]
    loss_repro = LossKeypointsMirror2DDirect(keypoints2d, bboxes, Pall, normal, cfg,
        mirror=params.pop('mirror', None))
    loss_funcs = {
        'k2d': loss_repro,
        'init_poses': LossInit(params, cfg).init_poses,
        'init_shapes': LossInit(params, cfg).init_shapes,
    }
    postprocess_funcs = [
        dict_of_tensor_to_numpy,
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, 
        extra_params=[loss_repro.mirror],
        weight_loss=weight, cfg=cfg)
    mirror = loss_repro.mirror.detach().cpu().numpy()
    params = flipSMPLParamsV(params, mirror)
    params['mirror'] = mirror
    return params

def viewSelection(params, body_model, loss_repro, nFrames):
    # view selection
    params_inp = {key: val.copy() for key, val in params.items()}
    params_inp = flipSMPLPosesV(params_inp)
    kpts_est = body_model(return_verts=False, return_tensor=True, **params_inp)
    residual = loss_repro.residual(kpts_est)
    res_i = torch.norm(residual, dim=-1).mean(dim=-1).sum(dim=0)
    params_rev = {key: val.copy() for key, val in params.items()}
    params_rev = flipSMPLPosesV(params_rev, reverse=True)
    kpts_est = body_model(return_verts=False, return_tensor=True, **params_rev)
    residual = loss_repro.residual(kpts_est)
    res_o = torch.norm(residual, dim=-1).mean(dim=-1).sum(dim=0)
    for nf in range(res_i.shape[0]):
        if res_i[nf] < res_o[nf]: # 使用外面的
            params['poses'][[nFrames+nf]] = flipSMPLPoses(params['poses'][[nf]])
        else:
            params['poses'][[nf]] = flipSMPLPoses(params['poses'][[nFrames+nf]])
    return params

def optimizeMirrorSoft(body_model, params, bboxes, keypoints2d, Pall, normal, weight, cfg):
    """ 
        simple function for optimizing mirror
        
    Args:
        body_model (SMPL model)
        params (DictParam): poses(2, 72), shapes(1, 10), Rh(2, 3), Th(2, 3)
        bboxes (nViews, nFrames, 5): 2D bbox of each view，输入的时候是按照时序叠起来的
        keypoints2d (nViews, nFrames, nJoints, 3): 2D keypoints of each view，输入的时候是按照时序叠起来的
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    nViews, nFrames = keypoints2d.shape[:2]
    assert nViews == 2, 'Please make sure that there exists only 2 views'
    prepare_funcs = [
        deepcopy_tensor,
        flipSMPLPosesV, #
        get_prepare_smplx(params, cfg, nFrames*nViews)
    ]
    loss_sym = LossMirrorSymmetry(normal=normal, cfg=cfg)
    loss_repro = LossKeypointsMirror2D(keypoints2d, bboxes, Pall, cfg)
    params = viewSelection(params, body_model, loss_repro, nFrames)
    init = LossInit(params, cfg)
    loss_funcs = {
        'k2d': loss_repro.__call__,
        'init_poses': init.init_poses,
        'init_shapes': init.init_shapes,
        'par_self': loss_sym.parallel_self,
        'ver_self': loss_sym.vertical_self,
        'par_mirror': loss_sym.parallel_mirror,
    }
    if nFrames > 1:
        loss_funcs['smooth_body'] = LossSmoothBodyMulti([0, nFrames, nFrames*2], cfg)
        loss_funcs['smooth_poses'] = LossSmoothPosesMulti([0, nFrames, nFrames*2], cfg)
    postprocess_funcs = [
        dict_of_tensor_to_numpy,
        flipSMPLPosesV
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight, cfg=cfg)
    return params