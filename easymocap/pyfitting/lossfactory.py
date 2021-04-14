'''
  @ Date: 2020-11-19 17:46:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-14 11:46:56
  @ FilePath: /EasyMocap/easymocap/pyfitting/lossfactory.py
'''
import numpy as np
import torch
from .operation import projection, batch_rodrigues

funcl2 = lambda x: torch.sum(x**2)
funcl1 = lambda x: torch.sum(torch.abs(x**2))

def gmof(squared_res, sigma_squared):
    """
    Geman-McClure error function
    """
    return (sigma_squared * squared_res) / (sigma_squared + squared_res)

def ReprojectionLoss(keypoints3d, keypoints2d, K, Rc, Tc, inv_bbox_sizes, norm='l2'):
    img_points = projection(keypoints3d, K, Rc, Tc)
    residual = (img_points - keypoints2d[:, :, :2]) * keypoints2d[:, :, -1:]
    # squared_res: (nFrames, nJoints, 2)
    if norm == 'l2':
        squared_res = (residual ** 2) * inv_bbox_sizes
    elif norm == 'l1':
        squared_res = torch.abs(residual) * inv_bbox_sizes
    else:
        import ipdb; ipdb.set_trace()
    return torch.sum(squared_res)

class LossKeypoints3D:
    def __init__(self, keypoints3d, cfg, norm='l2') -> None:
        self.cfg = cfg
        keypoints3d = torch.Tensor(keypoints3d).to(cfg.device)
        self.nJoints = keypoints3d.shape[1]
        self.keypoints3d = keypoints3d[..., :3]
        self.conf = keypoints3d[..., 3:]
        self.nFrames = keypoints3d.shape[0]
        self.norm = norm

    def loss(self, diff_square):
        if self.norm == 'l2':
            loss_3d = funcl2(diff_square)
        elif self.norm == 'l1':
            loss_3d = funcl1(diff_square)
        elif self.norm == 'gm':
            # 阈值设为0.2^2米
            loss_3d = torch.sum(gmof(diff_square**2, 0.04))
        else:
            raise NotImplementedError
        return loss_3d/self.nFrames

    def body(self, kpts_est, **kwargs):
        "distance of keypoints3d"
        nJoints = min([kpts_est.shape[1], self.keypoints3d.shape[1], 25])
        diff_square = (kpts_est[:, :nJoints, :3] - self.keypoints3d[:, :nJoints, :3])*self.conf[:, :nJoints]
        return self.loss(diff_square)

    def hand(self, kpts_est, **kwargs):
        "distance of 3d hand keypoints"
        diff_square = (kpts_est[:, 25:25+42, :3] - self.keypoints3d[:, 25:25+42, :3])*self.conf[:, 25:25+42]
        return self.loss(diff_square)
    
    def face(self, kpts_est, **kwargs):
        "distance of 3d face keypoints"
        diff_square = (kpts_est[:, 25+42:, :3] - self.keypoints3d[:, 25+42:, :3])*self.conf[:, 25+42:]
        return self.loss(diff_square)

    def __str__(self) -> str:
        return 'Loss function for keypoints3D, norm = {}'.format(self.norm)

class LossRegPoses:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def reg_hand(self, poses, **kwargs):
        "regulizer for hand pose"
        assert self.cfg.model in ['smplh', 'smplx']
        hand_poses = poses[:, 66:78]
        loss = funcl2(hand_poses)
        return loss/poses.shape[0]

    def reg_head(self, poses, **kwargs):
        "regulizer for head pose"
        assert self.cfg.model in ['smplx']
        poses = poses[:, 78:]
        loss = funcl2(poses)
        return loss/poses.shape[0]

    def reg_expr(self, expression, **kwargs):
        "regulizer for expression"
        assert self.cfg.model in ['smplh', 'smplx']
        return torch.sum(expression**2)

    def reg_body(self, poses, **kwargs):
        "regulizer for body poses"        
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, :66]
        loss = funcl2(poses)
        return loss/poses.shape[0]
    
    def __str__(self) -> str:
        return 'Loss function for Regulizer of Poses'

class LossRegPosesZero:
    def __init__(self, keypoints, cfg) -> None:
        model_type = cfg.model
        if keypoints.shape[-2] <= 15:
            use_feet = False
            use_head = False
        else:
            use_feet = keypoints[..., [19, 20, 21, 22, 23, 24], -1].sum() > 0.1
            use_head = keypoints[..., [15, 16, 17, 18], -1].sum() > 0.1
        if model_type == 'smpl':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]
        elif model_type == 'smplh':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        elif model_type == 'smplx':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        else:
            raise NotImplementedError
        if not use_feet:
            SMPL_JOINT_ZERO_IDX.extend([7, 8])
        if not use_head:
            SMPL_JOINT_ZERO_IDX.extend([12, 15])
        SMPL_POSES_ZERO_IDX = [[j for j in range(3*i, 3*i+3)] for i in SMPL_JOINT_ZERO_IDX]
        SMPL_POSES_ZERO_IDX = sum(SMPL_POSES_ZERO_IDX, [])
        # SMPL_POSES_ZERO_IDX.extend([36, 37, 38, 45, 46, 47])
        self.idx = SMPL_POSES_ZERO_IDX

    def __call__(self, poses, **kwargs):
        "regulizer for zero joints"
        return torch.sum(torch.abs(poses[:, self.idx]))/poses.shape[0]
    
    def __str__(self) -> str:
        return 'Loss function for Regulizer of Poses'

class LossSmoothBody:
    def __init__(self, cfg) -> None:
        self.norm = 'l2'

    def __call__(self, kpts_est, **kwargs):
        N_BODY = min(25, kpts_est.shape[1])
        assert kpts_est.shape[0] > 1, 'If you use smooth loss, it must be more than 1 frames'
        if self.norm == 'l2':
            loss = funcl2(kpts_est[:-1, :N_BODY] - kpts_est[1:, :N_BODY])
        else:
            loss = funcl1(kpts_est[:-1, :N_BODY] - kpts_est[1:, :N_BODY])
        return loss/kpts_est.shape[0]
    
    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'

class LossSmoothBodyMean:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def smooth(self, kpts_est, **kwargs):
        "smooth body"
        kpts_interp = kpts_est.clone().detach()
        kpts_interp[1:-1] = (kpts_interp[:-2] + kpts_interp[2:])/2
        loss = funcl2(kpts_est[1:-1] - kpts_interp[1:-1])
        return loss/(kpts_est.shape[0] - 2)
    
    def body(self, kpts_est, **kwargs):
        "smooth body"
        return self.smooth(kpts_est[:, :25])
    
    def hand(self, kpts_est, **kwargs):
        "smooth body"
        return self.smooth(kpts_est[:, 25:25+42])

    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'

class LossSmoothPoses:
    def __init__(self, nViews, nFrames, cfg=None) -> None:
        self.nViews = nViews
        self.nFrames = nFrames
        self.norm = 'l2'
        self.cfg = cfg

    def _poses(self, poses):
        "smooth poses"
        loss = 0
        for nv in range(self.nViews):
            poses_ = poses[nv*self.nFrames:(nv+1)*self.nFrames, ]
            # 计算poses插值
            poses_interp = poses_.clone().detach()
            poses_interp[1:-1] = (poses_interp[1:-1] + poses_interp[:-2] + poses_interp[2:])/3
            loss += funcl2(poses_[1:-1] - poses_interp[1:-1])
        return loss/(self.nFrames-2)/self.nViews
    
    def poses(self, poses, **kwargs):
        "smooth body poses"
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, :66]
        return self._poses(poses)
    
    def hands(self, poses, **kwargs):
        "smooth hand poses"
        if self.cfg.model in ['smplh', 'smplx']:
            poses = poses[:, 66:66+12]
        else:
            raise NotImplementedError
        return self._poses(poses)
    
    def head(self, poses, **kwargs):
        "smooth head poses"
        if self.cfg.model == 'smplx':
            poses = poses[:, 66+12:]
        else:
            raise NotImplementedError
        return self._poses(poses)

    def __str__(self) -> str:
        return 'Loss function for Smooth of Body'

class LossSmoothBodyMulti(LossSmoothBody):
    def __init__(self, dimGroups, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.dimGroups = dimGroups
    
    def __call__(self, kpts_est, **kwargs):
        "Smooth body"
        assert kpts_est.shape[0] > 1, 'If you use smooth loss, it must be more than 1 frames'
        loss = 0
        for nv in range(len(self.dimGroups) - 1):
            kpts = kpts_est[self.dimGroups[nv]:self.dimGroups[nv+1]]
            loss += super().__call__(kpts_est=kpts)
        return loss/(len(self.dimGroups) - 1)

    def __str__(self) -> str:
        return 'Loss function for Multi Smooth of Body'

class LossSmoothPosesMulti:
    def __init__(self, dimGroups, cfg) -> None:
        self.dimGroups = dimGroups
        self.norm = 'l2'
    
    def __call__(self, poses, **kwargs):
        "Smooth poses"
        loss = 0
        for nv in range(len(self.dimGroups) - 1):
            poses_ = poses[self.dimGroups[nv]:self.dimGroups[nv+1]]
            poses_interp = poses_.clone().detach()
            poses_interp[1:-1] = (poses_interp[1:-1] + poses_interp[:-2] + poses_interp[2:])/3
            loss += funcl2(poses_[1:-1] - poses_interp[1:-1])/(poses_.shape[0] - 2)
        return loss/(len(self.dimGroups) - 1)
    
    def __str__(self) -> str:
        return 'Loss function for Multi Smooth of Poses'
class LossRepro:
    def __init__(self, bboxes, keypoints2d, cfg) -> None:
        device = cfg.device
        bbox_sizes = np.maximum(bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1])
        # 这里的valid不是一维的，因为不清楚总共有多少维，所以不能遍历去做
        bbox_conf = bboxes[..., 4]
        bbox_mean_axis = -1
        bbox_sizes = (bbox_sizes * bbox_conf).sum(axis=bbox_mean_axis)/(1e-3 + bbox_conf.sum(axis=bbox_mean_axis))
        bbox_sizes = bbox_sizes[..., None, None, None]
        # 抑制掉完全不可见的视角，将其置信度设成0
        bbox_sizes[bbox_sizes < 10] = 1e6
        inv_bbox_sizes = torch.Tensor(1./bbox_sizes).to(device)
        keypoints2d = torch.Tensor(keypoints2d).to(device)
        self.keypoints2d = keypoints2d[..., :2]
        self.conf = keypoints2d[..., 2:] * inv_bbox_sizes * 100
        self.norm = 'gm'
    
    def __call__(self, img_points):
        residual = (img_points - self.keypoints2d) * self.conf
        # squared_res: (nFrames, nJoints, 2)
        if self.norm == 'l2':
            squared_res = residual ** 2
        elif self.norm == 'l1':
            squared_res = torch.abs(residual)
        elif self.norm == 'gm':
            squared_res = gmof(residual**2, 200)
        else:
            import ipdb; ipdb.set_trace()
        return torch.sum(squared_res)

class LossInit:
    def __init__(self, params, cfg) -> None:
        self.norm = 'l2'
        self.poses = torch.Tensor(params['poses']).to(cfg.device)
        self.shapes = torch.Tensor(params['shapes']).to(cfg.device)

    def init_poses(self, poses, **kwargs):
        "distance to poses_0"
        if self.norm == 'l2':
            return torch.sum((poses - self.poses)**2)/poses.shape[0]
    
    def init_shapes(self, shapes, **kwargs):
        "distance to shapes_0"
        if self.norm == 'l2':
            return torch.sum((shapes - self.shapes)**2)/shapes.shape[0]

class LossKeypointsMV2D(LossRepro):
    def __init__(self, keypoints2d, bboxes, Pall, cfg) -> None:
        """
        Args:
            keypoints2d (ndarray): (nViews, nFrames, nJoints, 3)
            bboxes (ndarray): (nViews, nFrames, 5)
        """
        super().__init__(bboxes, keypoints2d, cfg)
        assert Pall.shape[0] == keypoints2d.shape[0] and Pall.shape[0] == bboxes.shape[0], \
            'check you P shape: {} and keypoints2d shape: {}'.format(Pall.shape, keypoints2d.shape)
        device = cfg.device
        self.Pall = torch.Tensor(Pall).to(device)
        self.nViews, self.nFrames, self.nJoints = keypoints2d.shape[:3]
        self.kpt_homo = torch.ones((self.nFrames, self.nJoints, 1), device=device)

    def __call__(self, kpts_est, **kwargs):
        "reprojection loss for multiple views"
        # kpts_est: (nFrames, nJoints, 3+1), P: (nViews, 3, 4)
        #   => projection: (nViews, nFrames, nJoints, 3)
        kpts_homo = torch.cat([kpts_est[..., :self.nJoints, :], self.kpt_homo], dim=2)
        point_cam = torch.einsum('vab,fnb->vfna', self.Pall, kpts_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        return super().__call__(img_points)/self.nViews/self.nFrames
    
    def __str__(self) -> str:
        return 'Loss function for Reprojection error'

class SMPLAngleLoss:
    def __init__(self, keypoints, model_type='smpl'):
        if keypoints.shape[1] <= 15:
            use_feet = False
            use_head = False
        else:
            use_feet = keypoints[:, [19, 20, 21, 22, 23, 24], -1].sum() > 0.1
            use_head = keypoints[:, [15, 16, 17, 18], -1].sum() > 0.1
        if model_type == 'smpl':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]
        elif model_type == 'smplh':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        elif model_type == 'smplx':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        else:
            raise NotImplementedError
        if not use_feet:
            SMPL_JOINT_ZERO_IDX.extend([7, 8])
        if not use_head:
            SMPL_JOINT_ZERO_IDX.extend([12, 15])
        SMPL_POSES_ZERO_IDX = [[j for j in range(3*i, 3*i+3)] for i in SMPL_JOINT_ZERO_IDX]
        SMPL_POSES_ZERO_IDX = sum(SMPL_POSES_ZERO_IDX, [])
        # SMPL_POSES_ZERO_IDX.extend([36, 37, 38, 45, 46, 47])
        self.idx = SMPL_POSES_ZERO_IDX

    def loss(self, poses):
        return torch.sum(torch.abs(poses[:, self.idx]))

def SmoothLoss(body_params, keys, weight_loss, span=4, model_type='smpl'):
    spans = [i for i in range(1, span)]
    span_weights = {i:1/i for i in range(1, span)}
    span_weights = {key: i/sum(span_weights) for key, i in span_weights.items()}
    loss_dict = {}
    nFrames = body_params['poses'].shape[0]
    nPoses = body_params['poses'].shape[1]
    if model_type == 'smplh' or model_type == 'smplx':
        nPoses = 66
    for key in ['poses', 'Th', 'poses_hand', 'expression']:
        if key not in keys:
            continue
        k = 'smooth_' + key
        if k in weight_loss.keys() and weight_loss[k] > 0.:
            loss_dict[k] = 0.
            for span in spans:
                if key == 'poses_hand':
                    val = torch.sum((body_params['poses'][span:, 66:] - body_params['poses'][:nFrames-span, 66:])**2)
                else:
                    val = torch.sum((body_params[key][span:, :nPoses] - body_params[key][:nFrames-span, :nPoses])**2)
                loss_dict[k] += span_weights[span] * val
        k = 'smooth_' + key + '_l1'
        if k in weight_loss.keys() and weight_loss[k] > 0.:
            loss_dict[k] = 0.
            for span in spans:
                if key == 'poses_hand':
                    val = torch.sum((body_params['poses'][span:, 66:] - body_params['poses'][:nFrames-span, 66:]).abs())
                else:
                    val = torch.sum((body_params[key][span:, :nPoses] - body_params[key][:nFrames-span, :nPoses]).abs())
                loss_dict[k] += span_weights[span] * val
    # smooth rotation
    rot = batch_rodrigues(body_params['Rh'])
    key, k = 'Rh', 'smooth_Rh'
    if key in keys and k in weight_loss.keys() and weight_loss[k] > 0.:
        loss_dict[k] = 0.
        for span in spans:
            val = torch.sum((rot[span:, :] - rot[:nFrames-span, :])**2)
            loss_dict[k] += span_weights[span] * val
    return loss_dict

def RegularizationLoss(body_params, body_params_init, weight_loss):
    loss_dict = {}
    for key in ['poses', 'shapes', 'Th', 'hands', 'head', 'expression']:
        if 'init_'+key in weight_loss.keys() and weight_loss['init_'+key] > 0.:
            if key == 'poses':
                loss_dict['init_'+key] = torch.sum((body_params[key][:, :66] - body_params_init[key][:, :66])**2)
            elif key == 'hands':
                loss_dict['init_'+key] = torch.sum((body_params['poses'][: , 66:66+12] - body_params_init['poses'][:, 66:66+12])**2)
            elif key == 'head':
                loss_dict['init_'+key] = torch.sum((body_params['poses'][: , 78:78+9] - body_params_init['poses'][:, 78:78+9])**2)
            elif key in body_params.keys():
                loss_dict['init_'+key] = torch.sum((body_params[key] - body_params_init[key])**2)
    for key in ['poses', 'shapes', 'hands', 'head', 'expression']:
        if 'reg_'+key in weight_loss.keys() and weight_loss['reg_'+key] > 0.:
            if key == 'poses':
                loss_dict['reg_'+key] = torch.sum((body_params[key][:, :66])**2)
            elif key == 'hands':
                loss_dict['reg_'+key] = torch.sum((body_params['poses'][: , 66:66+12])**2)
            elif key == 'head':
                loss_dict['reg_'+key] = torch.sum((body_params['poses'][: , 78:78+9])**2)
            elif key in body_params.keys():
                loss_dict['reg_'+key] = torch.sum((body_params[key])**2)
    return loss_dict