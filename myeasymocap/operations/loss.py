import torch
import torch.nn as nn
import numpy as np

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho2 = rho * rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, est, gt=None, conf=None):
        if gt is not None:
            square_diff = torch.sum((est - gt)**2, dim=-1)
        else:
            square_diff = torch.sum(est**2, dim=-1)
        diff = torch.div(square_diff, square_diff + self.rho2)
        if conf is not None:
            res = torch.sum(diff * conf)/(1e-5 + conf.sum())
        else:
            res = diff.sum()/diff.numel()
        return res

class BaseLoss(nn.Module):
    def __init__(self, norm='l2', norm_info={}, reduce='sum') -> None:
        super().__init__()
        self.loss = self.make_loss(norm, norm_info, reduce)
    
    def make_loss(self, norm='l2', norm_info={}, reduce='sum'):
        reduce = torch.sum if reduce=='sum' else torch.mean
        if norm == 'l2':
            def loss(est, gt=None, conf=None):
                if gt is not None:
                    square_diff = reduce((est - gt)**2, dim=-1)
                else:
                    square_diff = reduce(est**2, dim=-1)
                if conf is not None:
                    res = torch.sum(square_diff * conf)/(1e-5 + conf.sum())
                else:
                    res = square_diff.sum()/square_diff.numel()
                return res
        elif norm == 'l1':
            def loss(est, gt=None, conf=None):
                if gt is not None:
                    square_diff = reduce(torch.abs(est - gt), dim=-1)
                else:
                    square_diff = reduce(torch.abs(est), dim=-1)
                if conf is not None:
                    res = torch.sum(square_diff * conf)/(1e-5 + conf.sum())
                else:
                    res = square_diff.sum()/square_diff.numel()
                return res
        elif norm == 'gm':
            loss = GMoF(norm_info)
        else:
            loss = None
        return loss

    def forward(self, pred, target):
        pass

class BaseKeypoints(BaseLoss):
    @staticmethod
    def select(keypoints, index, ranges):
        if len(index) > 0:
            keypoints = keypoints[..., index, :]
        elif len(ranges) > 0:
            if ranges[1] == -1:
                keypoints = keypoints[..., ranges[0]:, :]
            else:
                keypoints = keypoints[..., ranges[0]:ranges[1], :]
        return keypoints

    def __init__(self, index_est=[], index_gt=[],
                ranges_est=[], ranges_gt=[], **kwargs):
        super().__init__(**kwargs)
        self.index_est = index_est
        self.index_gt = index_gt
        self.ranges_est = ranges_est
        self.ranges_gt = ranges_gt

    def forward(self, pred, target):
        return super().forward(pred, target)
    
    def loss_keypoints(self, pred, target, conf):
        # pred: (..., dim)
        # target: (..., dim)
        # conf: (..., 1)
        dist = torch.sum((pred - target)**2, dim=-1, keepdim=True)
        loss = torch.sum(dist * conf) / torch.sum(conf)
        return loss

class Keypoints2D(BaseKeypoints):
    def forward(self, pred, target):
        # (nFrames, nJoints, 3)
        pred_kpts3d = self.select(pred['keypoints'] , self.index_est, self.ranges_est)
        target_kpts2d = self.select(target['keypoints'], self.index_gt, self.ranges_gt)
        cameras = target['cameras']
        P = torch.cat([cameras['R'], cameras['T']], dim=-1)
        invKtrans = torch.inverse(cameras['K']).transpose(-1, -2)
        homo = torch.cat([target_kpts2d[..., :2], torch.ones_like(target_kpts2d[..., 2:])], dim=-1)
        target_points = torch.matmul(homo, invKtrans)[..., :2]
        pred_homo = torch.cat([pred_kpts3d, torch.ones_like(pred_kpts3d[..., :1])], dim=-1)
        self.einsum = 'fab,fjb->fja'
        point_cam = torch.einsum(self.einsum, P, pred_homo)
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        loss = self.loss(est=img_points, gt=target_points, conf=target_kpts2d[..., -1])
        return loss
    
class Keypoints3D(BaseKeypoints):
    def forward(self, pred, target):
        # (nFrames, nJoints, 3)
        # breakpoint()
        pred_kpts3d = self.select(pred['keypoints'] , self.index_est, self.ranges_est)
        target_kpts3d = self.select(target['keypoints3d'], self.index_gt, self.ranges_gt)
        assert target_kpts3d.shape[-1] == 4, 'Target keypoints {} must have confidence '.format(target_kpts3d.shape)
        loss = self.loss(est=pred_kpts3d, gt=target_kpts3d[...,:3], conf=target_kpts3d[..., -1])
        return loss

class LimbLength(BaseKeypoints):
    def __init__(self, kintree, key='keypoints3d', **kwargs):
        self.kintree = np.array(kintree)
        super().__init__(**kwargs)
    
    def __str__(self):
        return "Limb of: {}".format(','.join(['[{},{}]'.format(i,j) for (i,j) in self.kintree]))

    def forward(self, pred, target):
        pred_kpts3d = pred['keypoints']
        target_kpts3d = target['keypoints3d']
        # 用kin tree来进行选择
        pred = torch.norm(pred_kpts3d[..., self.kintree[:, 1], :] - pred_kpts3d[..., self.kintree[:, 0], :], dim=-1, keepdim=True)
        target = torch.norm(target_kpts3d[..., self.kintree[:, 1], :] - target_kpts3d[..., self.kintree[:, 0], :], dim=-1, keepdim=True)
        target_conf = torch.minimum(target_kpts3d[..., self.kintree[:, 1], -1], target_kpts3d[..., self.kintree[:, 0], -1])
        loss = self.loss(est=pred, gt=target, conf=target_conf)
        return loss

class Smooth(BaseLoss):
    def __init__(self, keys, smooth_type, order, norm, weights, window_weight) -> None:
        super().__init__(norm)
        self.loss = {}
        for i in range(len(keys)):
            new_key = keys[i] + '_' + smooth_type[i]
            self.loss[new_key] = {
                'func': self.make_loss(norm='l2', norm_info={}, reduce='sum'),
                'key': keys[i],
                'weight': weights[i],
                'norm': norm[i],
                'order': order[i],
                'type': smooth_type[i],
            }
        self.window_weight = window_weight
    
    def convert_Rh_to_R(self, Rh):
        from ..bodymodels.geometry import batch_rodrigues
        # Rh: (..., nRot x 3)
        nRot = Rh.shape[-1] // 3
        Rh_flat = Rh.reshape(-1, nRot, 3)
        Rh_flat = Rh_flat.reshape(-1, 3)
        Rot = batch_rodrigues(Rh_flat)
        Rot_0 = Rot.reshape(-1, nRot, 3, 3)
        Rot = Rot_0.reshape(*Rh.shape[:-1], 3, 3)
        Rot = Rot.reshape(*Rh.shape[:-1], 9)
        return Rot
    
    def forward(self, pred, target):
        ret = {}
        for key, cfg in self.loss.items():
            value = pred[cfg['key']]
            loss = 0
            for width, weight in enumerate(self.window_weight, start=1):
                if cfg['type'] == 'Linear':
                    vel = value[width:] - value[:-width]
                elif cfg['type'] == 'Rot':
                    _value = self.convert_Rh_to_R(value)
                    vel = _value[width:] - _value[:-width]
                elif cfg['type'] == 'Depth':
                    # TODO: 考虑相机的RT
                    if 'cameras' in target.keys():
                        R = target['cameras']['R']
                        _value = torch.bmm(value[..., None, :], R.transpose(-1, -2))
                        _value = _value[..., 0, :]
                    _value = _value[..., [2]] # 只使用深度
                    vel = _value[width:] - _value[:-width]
                if cfg['order'] == 2:
                    vel = vel[1:] - vel[:-1]
                loss += weight * cfg['func'](est=vel)
            ret[key] = loss * cfg['weight']
        return ret

class AnySmooth(BaseLoss):
    def __init__(self, key, weight, norm, norm_info={}, dim=-1, order=1):
        super().__init__()
        self.dim = dim
        self.weight = weight
        self.loss = self.make_loss(norm, norm_info)
        self.norm_name = norm
        self.key = key
        self.order = order
    
    def forward(self, pred, target):
        loss = 0
        value = pred[self.key]
        # value = select(value, self.ranges, self.index, self.dim)
        if value.shape[0] <= len(self.weight):
            return torch.FloatTensor([0.]).to(value.device)
        for width, weight in enumerate(self.weight, start=1):
            vel = value[width:] - value[:-width]
            if self.order == 2:
                vel = vel[1:] - vel[:-1]
            loss += weight * self.loss(vel)
        return loss

class Init(BaseLoss):
    def __init__(self, keys, weights, norm) -> None:
        super().__init__(norm)
        self.keys = keys
        self.weights = weights

    def forward(self, pred, target):
        ret = {}
        for key in self.keys:
            ret[key] = torch.mean((pred[key] - target['init_'+key])**2)
        return ret

from easymocap.multistage.lossbase import AnyReg
class RegLoss(AnyReg):
    def __init__(self, key, norm) -> None:
        super().__init__(key, norm)

    def __call__(self, pred, target):
        return self.forward(**{self.key: pred[self.key]})
    
class Init_pose(Init):
    def __init__(self, keys, weights, norm) -> None:
        super().__init__(keys, weights, norm)
        self.norm = norm
    def forward(self, pred, target):
        ret = {}
        for key in self.keys:
            if self.norm == 'l2':
                ret[key] = torch.sum((pred[key] - target['target_'+key])**2)
            elif self.norm == 'l1':
                ret[key] = torch.sum(torch.abs(pred[key] - target['target_'+key]))
        return ret