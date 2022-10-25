import numpy as np
import torch.nn as nn
import torch
from ..bodymodel.lbs import batch_rodrigues

class LossBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def __str__(self) -> str:
        return '# lack of comment'
    
    def check_at_start(self, **kwargs):
        pass
    
    def check_at_end(self, **kwargs):
        pass

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

def make_loss(norm, norm_info, reduce='sum'):
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
    elif norm == 'gm':
        loss = GMoF(norm_info)
    return loss

def select(value, ranges, index, dim):
    if len(ranges) > 0:
        if ranges[1] == -1:
            value = value[..., ranges[0]:]
        else:
            value = value[..., ranges[0]:ranges[1]]
        return value
    if len(index) > 0:
        if dim == -1:
            value = value[..., index]
        elif dim == -2:
            value = value[..., index, :]
        return value
    return value

def print_table(header, contents):
    from tabulate import tabulate
    length = len(contents[0])
    tables = [[] for _ in range(length)]
    mean = ['Mean']
    for icnt, content in enumerate(contents):
        for i in range(length):
            if isinstance(content[i], float):
                tables[i].append('{:6.2f}'.format(content[i]))
            else:
                tables[i].append('{}'.format(content[i]))
        if icnt > 0:
            mean.append('{:6.2f}'.format(sum(content)/length))
    tables.append(mean)
    print(tabulate(tables, header, tablefmt='fancy_grid'))

class AnyReg(LossBase):
    def __init__(self, key, norm, dim=-1, reduce='sum', norm_info={}, ranges=[], index=[], **kwargs):
        super().__init__()
        self.ranges = ranges
        self.index = index
        self.key = key
        self.dim = dim
        if 'init_' + key in kwargs.keys():
            init = kwargs['init_'+key]
            self.register_buffer('init', torch.Tensor(init))
        else:
            self.init = None
        self.norm_name = norm
        self.loss = make_loss(norm, norm_info, reduce=reduce)

    def forward(self, **kwargs):
        """
            value: (nFrames, ..., nDims)
        """
        value = kwargs[self.key]
        if self.init is not None:
            value = value - self.init
        value = select(value, self.ranges, self.index, self.dim)
        return self.loss(value)
    
    def __str__(self) -> str:
        return 'Loss for {}'.format(self.key, self.norm_name)

class RegPrior(AnyReg):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.init = None # disable init
        infos = {
            (2, 0): '-exp',
            (2, 1): 'l2',
            (2, 2): 'l2',
            (3, 0): '-exp', # knee
            (3, 1): 'L2',
            (3, 2): 'L2',
            (4, 0): '-exp', # knee
            (4, 1): 'L2',
            (4, 2): 'L2',
            (5, 0): '-exp',
            (5, 1): 'l2',
            (5, 2): 'l2',
            (6, 1): 'L2',
            (6, 2): 'L2',
            (7, 1): 'L2',
            (7, 2): 'L2',
            (8, 0): '-exp',
            (8, 1): 'l2',
            (8, 2): 'l2',
            (9, 0): 'L2',
            (9, 1): 'L2',
            (9, 2): 'L2',
            (10, 0): 'L2',
            (10, 1): 'L2',
            (10, 2): 'L2',
            (12, 0): 'l2', # 肩关节前面
            (13, 0): 'l2',
            (17, 1): 'exp',
            (17, 2): 'L2',
            (18, 1): '-exp',
            (18, 2): 'L2',
        }
        self.l2dims = []
        self.L2dims = []
        self.expdims = []
        self.nexpdims = []
        for (nj, ndim), norm in infos.items():
            dim = nj*3 + ndim
            if norm == 'l2':
                self.l2dims.append(dim)
            elif norm == 'L2':
                self.L2dims.append(dim)
            elif norm == '-exp':
                self.nexpdims.append(dim)
            elif norm == 'exp':
                self.expdims.append(dim)

    def forward(self, poses, **kwargs):
        """
            poses: (..., nDims)
        """
        alll2loss = torch.mean(poses**2)
        l2loss = torch.sum(poses[:, self.l2dims]**2)/len(self.l2dims)/poses.shape[0]
        L2loss = torch.sum(poses[:, self.L2dims]**2)/len(self.L2dims)/poses.shape[0]
        exploss = torch.sum(torch.exp(poses[:, self.expdims]))/poses.shape[0]
        nexploss = torch.sum(torch.exp(-poses[:, self.nexpdims]))/poses.shape[0]
        loss = 0.1*l2loss + L2loss + 0.0005*(exploss + nexploss)/(len(self.expdims) + len(self.nexpdims)) + 0.01*alll2loss
        return loss

class VPoserPrior(AnyReg):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        vposer_ckpt = 'data/bodymodels/vposer_v02'
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        vposer, _ = load_model(vposer_ckpt, 
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
        vposer.eval()
        self.vposer = vposer
        self.init = None # disable init
    
    def forward(self, poses, **kwargs):
        """
            poses: (..., nDims)
        """
        nDims = 63
        poses_body = poses[..., :nDims].reshape(-1, nDims)
        latent = self.vposer.encode(poses_body)
        if True:
            ret = self.vposer.decode(latent.sample())['pose_body'].reshape(poses.shape[0], nDims)
            return super().forward(poses=poses_body-ret)
        else:
            return super().forward(poses=latent.mean)

class AnySmooth(LossBase):
    def __init__(self, key, weight, norm, norm_info={}, ranges=[], index=[], dim=-1, order=1):
        super().__init__()
        self.ranges = ranges
        self.index = index
        self.dim = dim
        self.weight = weight
        self.loss = make_loss(norm, norm_info)
        self.norm_name = norm
        self.key = key
        self.order = order
    
    def forward(self, **kwargs):
        loss = 0
        value = kwargs[self.key]
        value = select(value, self.ranges, self.index, self.dim)
        if value.shape[0] <= len(self.weight):
            return torch.FloatTensor([0.]).to(value.device)
        for width, weight in enumerate(self.weight, start=1):
            vel = value[width:] - value[:-width]
            if self.order == 2:
                vel = vel[1:] - vel[:-1]
            loss += weight * self.loss(vel)
        return loss

    def check(self, value):
        vel = value[1:] - value[:-1]
        if len(vel.shape) > 2:
            vel = torch.norm(vel, dim=-1)
        else:
            vel = torch.abs(vel)
        vel = vel.detach().cpu().numpy()
        return vel

    def get_check_name(self, value):
        name = [str(i) for i in range(value.shape[1])]
        return name

    def check_at_start(self, **kwargs):
        if self.key not in kwargs.keys():
            return 0
        value = kwargs[self.key]
        if value.shape[0] < len(self.weight):
            return 0
        header = ['Smooth '+self.key, 'mean(before)', 'max(before)', 'frame(before)']
        name = self.get_check_name(value)
        vel = self.check(value)
        contents = [name, vel.mean(axis=0).tolist(), vel.max(axis=0).tolist(), vel.argmax(axis=0).tolist()]
        self.cache_check = (header, contents)
        return super().check_at_start(**kwargs)

    def check_at_end(self, **kwargs):
        if self.key not in kwargs.keys():
            return 0
        value = kwargs[self.key]
        if value.shape[0] < len(self.weight):
            return 0
        err_after = self.check(kwargs[self.key])
        header, contents = self.cache_check
        header.extend(['mean(after)', 'max(after)', 'frame(after)'])
        contents.extend([err_after.mean(axis=0).tolist(), err_after.max(axis=0).tolist(), err_after.argmax(axis=0).tolist()])
        print_table(header, contents)

    def __str__(self) -> str:
        return "smooth in {} frames, range={}, norm={}".format(self.weight, self.ranges, self.norm_name)

class SmoothRot(AnySmooth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ..bodymodel.lbs import batch_rodrigues
        self.rodrigues = batch_rodrigues

    def convert_Rh_to_R(self, Rh):
        shape = Rh.shape[1]
        ret = []
        for i in range(shape//3):
            Rot = self.rodrigues(Rh[:, 3*i:3*(i+1)])
            ret.append(Rot)
        ret = torch.cat(ret, dim=1)
        return ret

    def forward(self, **kwargs):
        Rh = kwargs[self.key]
        if Rh.shape[-1] != 3:
            loss = 0
            for i in range(Rh.shape[-1]//3):
                Rh_sub = Rh[..., 3*i:3*i+3]
                Rot = self.convert_Rh_to_R(Rh_sub).view(*Rh_sub.shape[:-1], 3, 3)
                loss += super().forward(**{self.key: Rot})
            return loss
        else:
            Rh_flat = Rh.view(-1, 3)
            Rot = self.convert_Rh_to_R(Rh_flat).view(*Rh.shape[:-1], 3, 3)
            return super().forward(**{self.key: Rot})
    
    def get_check_name(self, value):
        name = ['angle']
        return name

    def check(self, value):
        import cv2
        # TODO: here just use first rotation
        if len(value.shape) == 3:
            value = value[:, 0]
        Rot = self.convert_Rh_to_R(value.detach())[:, :3]
        vel = torch.matmul(Rot[:-1], Rot.transpose(1,2)[1:]).cpu().numpy()
        vels = []
        for i in range(vel.shape[0]):
            angle = np.linalg.norm(cv2.Rodrigues(vel[i])[0])
            vels.append(angle)
        vel = np.array(vels).reshape(-1, 1)
        return vel

class BaseKeypoints(LossBase):
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

    def set_gt(self, index_gt, ranges_gt):
        keypoints = self.select(self.keypoints_np, index_gt, ranges_gt)
        keypoints = torch.Tensor(keypoints)
        self.register_buffer('keypoints', keypoints[..., :-1])
        self.register_buffer('conf', keypoints[..., -1])

    def __str__(self):
        return "keypoints: {}".format(self.keypoints.shape)

    def __init__(self, keypoints, norm='l2', norm_info={},
        index_gt=[], ranges_gt=[],
        index_est=[], ranges_est=[]) -> None:
        super().__init__()
        # prepare ground-truth
        self.keypoints_np = keypoints
        self.set_gt(index_gt, ranges_gt)
        # 
        self.index_est = index_est
        self.ranges_est = ranges_est

        self.norm = norm
        self.loss = make_loss(norm, norm_info)
    
    def forward(self, kpts_est, **kwargs):
        est = self.select(kpts_est, self.index_est, self.ranges_est)
        return self.loss(est, self.keypoints, self.conf)

    def check(self, kpts_est, min_conf=0.3, **kwargs):
        est = self.select(kpts_est, self.index_est, self.ranges_est)
        conf = (self.conf>min_conf).float()
        norm = torch.norm(est-self.keypoints, dim=-1) * conf
        mean_joints = norm.sum(dim=0)/(1e-5 + conf.sum(dim=0)) * 1000
        return conf, mean_joints

    def check_at_start(self, kpts_est, **kwargs):
        if len(self.index_est) > 0:
            names = [str(i) for i in self.index_est]
        elif len(self.ranges_est) > 0:
            names = [str(i) for i in range(self.ranges_est[0], self.ranges_est[1])]
        else:
            names = [str(i) for i in range(self.conf.shape[-1])]
        conf, error = self.check(kpts_est, **kwargs)
        valid = conf.sum(dim=0).detach().cpu().numpy().tolist()
        header = ['name', 'count']
        contents = [names, valid]
        header.append('before')
        contents.append(error.detach().cpu().numpy().tolist())
        self.cache_check = (header, contents)
    
    def check_at_end(self, kpts_est, **kwargs):
        conf, err_after = self.check(kpts_est, **kwargs)
        header, contents = self.cache_check
        header.append('after')
        contents.append(err_after.detach().cpu().numpy().tolist())
        print_table(header, contents)

class Keypoints3D(BaseKeypoints):
    def __init__(self, keypoints3d, **kwargs) -> None:
        super().__init__(keypoints3d, **kwargs)

class AnyKeypoints3D(Keypoints3D):
    def __init__(self, **kwargs) -> None:
        key = kwargs.pop('key')
        keypoints3d = kwargs.pop(key)
        super().__init__(keypoints3d, **kwargs)
        self.key = key

class AnyKeypoints3DWithRT(Keypoints3D):
    def __init__(self, **kwargs) -> None:
        key = kwargs.pop('key')
        keypoints3d = kwargs.pop(key)
        super().__init__(keypoints3d, **kwargs)
        self.key = key
    
    def forward(self, kpts_est, **kwargs):
        R = batch_rodrigues(kwargs['R_'+self.key])
        T = kwargs['T_'+self.key]
        RXT = torch.matmul(kpts_est, R.transpose(-1, -2)) + T[..., None, :]
        return super().forward(RXT)

    def check(self, kpts_est, min_conf=0.3, **kwargs):
        R = batch_rodrigues(kwargs['R_'+self.key])
        T = kwargs['T_'+self.key]
        RXT = torch.matmul(kpts_est, R.transpose(-1, -2)) + T[..., None, :]
        kpts_est = RXT
        return super().check(kpts_est, min_conf)

class Handl3D(BaseKeypoints):
    def __init__(self, handl3d, **kwargs) -> None:
        handl3d = handl3d.clone()
        handl3d[..., :3] = handl3d[..., :3] - handl3d[:, :1, :3]
        super().__init__(handl3d, **kwargs)
    
    def forward(self, kpts_est, **kwargs):
        est = kpts_est[:, 25:46]
        est = est - est[:, :1].detach()
        return super().forward(est, **kwargs)
    
    def check(self, kpts_est, **kwargs):
        est = kpts_est[:, 25:46]
        est = est - est[:, :1].detach()
        return super().check(est, **kwargs)
    

class LimbLength(BaseKeypoints):
    def __init__(self, kintree, key='keypoints3d', **kwargs):
        self.kintree = np.array(kintree)
        if key == 'bodyhand':
            keypoints3d = np.hstack([kwargs.pop('keypoints3d'), kwargs.pop('handl3d'), kwargs.pop('handr3d')])
        else:
            keypoints3d = kwargs.pop(key)
        super().__init__(keypoints3d, **kwargs)
    
    def __str__(self):
        return "Limb of: {}".format(','.join(['[{},{}]'.format(i,j) for (i,j) in self.kintree]))

    def set_gt(self, index_gt, ranges_gt):
        keypoints3d = self.keypoints_np
        kintree = self.kintree
        # limb_length: nFrames, nLimbs, 1
        limb_length = np.linalg.norm(keypoints3d[..., kintree[:, 1], :3] - keypoints3d[..., kintree[:, 0], :3], axis=-1, keepdims=True)
        # conf: nFrames, nLimbs, 1
        limb_conf = np.minimum(keypoints3d[..., kintree[:, 1], -1], keypoints3d[..., kintree[:, 0], -1])
        limb_length = torch.Tensor(limb_length)
        limb_conf = torch.Tensor(limb_conf)
        self.register_buffer('length', limb_length)
        self.register_buffer('conf', limb_conf)

    def forward(self, kpts_est, **kwargs):
        src = kpts_est[..., self.kintree[:, 0], :]
        dst = kpts_est[..., self.kintree[:, 1], :]
        length_est = torch.norm(dst - src, dim=-1, keepdim=True)
        return self.loss(length_est, self.length, self.conf)

    def check_at_start(self, kpts_est, **kwargs):
        names = [str(i) for i in self.kintree]
        conf = (self.conf>0)
        valid = conf.sum(dim=0).detach().cpu().numpy()
        if len(valid.shape) == 2:
            valid = valid.mean(axis=0)
        header = ['name', 'count']
        contents = [names, valid.tolist()]
        error, length = self.check(kpts_est)
        header.append('before')
        contents.append(error.detach().cpu().numpy().tolist())
        header.append('length')
        length = (self.length[..., 0] * self.conf).sum(dim=0)/self.conf.sum(dim=0)
        contents.append(length.detach().cpu().numpy().tolist())
        self.cache_check = (header, contents)
    
    def check_at_end(self, kpts_est, **kwargs):
        err_after, length = self.check(kpts_est)
        header, contents = self.cache_check
        header.append('after')
        contents.append(err_after.detach().cpu().numpy().tolist())
        header.append('length_est')
        contents.append(length[:,:,0].mean(dim=0).detach().cpu().numpy().tolist())
        print_table(header, contents)

    def check(self, kpts_est, **kwargs):
        src = kpts_est[..., self.kintree[:, 0], :]
        dst = kpts_est[..., self.kintree[:, 1], :]
        length_est = torch.norm(dst - src, dim=-1, keepdim=True)
        conf = (self.conf>0).float()
        norm = torch.abs(length_est-self.length)[..., 0] * conf
        mean_joints = norm.sum(dim=0)/conf.sum(dim=0) * 1000
        if len(mean_joints.shape) == 2:
            mean_joints = mean_joints.mean(dim=0)
            length_est = length_est.mean(dim=0)
        return mean_joints, length_est

class LimbLengthHand(LimbLength):
    def __init__(self, handl3d, handr3d, **kwargs):
        kintree = kwargs.pop('kintree')
        keypoints3d = torch.cat([handl3d, handr3d], dim=0)
        super().__init__(kintree, keypoints3d, **kwargs)
    
    def forward(self, kpts_est, **kwargs):
        kpts_est = torch.cat([kpts_est[:, :21], kpts_est[:, 21:]], dim=0)
        return super().forward(kpts_est, **kwargs)
    
    def check(self, kpts_est, **kwargs):
        kpts_est = torch.cat([kpts_est[:, :21], kpts_est[:, 21:]], dim=0)
        return super().check(kpts_est, **kwargs)

class Keypoints2D(BaseKeypoints):
    def __init__(self, keypoints2d, K, Rc, Tc, einsum='fab,fnb->fna', 
        unproj=True, reshape_views=False, **kwargs) -> None:
        # convert to camera coordinate
        invKtrans = torch.inverse(K).transpose(-1, -2)
        if unproj:
            homo = torch.ones_like(keypoints2d[..., :1])
            homo = torch.cat([keypoints2d[..., :2], homo], dim=-1)
            if len(invKtrans.shape) < len(homo.shape):
                invKtrans = invKtrans.unsqueeze(-3)
            homo = torch.matmul(homo, invKtrans)
            keypoints2d = torch.cat([homo[..., :2], keypoints2d[..., 2:]], dim=-1)
        # keypoints2d: (nFrames, nViews, ..., nJoints, 3)
        super().__init__(keypoints2d, **kwargs)
        self.register_buffer('K', K)
        self.register_buffer('invKtrans', invKtrans)
        self.register_buffer('Rc', Rc)
        self.register_buffer('Tc', Tc)
        self.unproj = unproj
        self.einsum = einsum
        self.reshape_views = reshape_views
    
    def project(self, kpts_est):
        kpts_est = self.select(kpts_est, self.index_est, self.ranges_est)
        kpts_homo = torch.ones_like(kpts_est[..., -1:])
        kpts_homo = torch.cat([kpts_est, kpts_homo], dim=-1)
        if self.unproj:
            P = torch.cat([self.Rc, self.Tc], dim=-1)
        else:
            P = torch.bmm(self.K, torch.cat([self.Rc, self.Tc], dim=-1))
        if self.reshape_views:
            kpts_homo = kpts_homo.reshape(self.K.shape[0], self.K.shape[1], *kpts_homo.shape[1:])
        try:
            point_cam = torch.einsum(self.einsum, P, kpts_homo)
        except:
            print('Wrong shape: {}x{} <=== {}'.format(P.shape, kpts_homo.shape, self.einsum))
            raise NotImplementedError
        img_points = point_cam[..., :2]/point_cam[..., 2:]
        return img_points

    def forward(self, kpts_est, **kwargs):
        img_points = self.project(kpts_est)
        loss = self.loss(img_points.squeeze(), self.keypoints.squeeze(), self.conf.squeeze())
        return loss
    
    def check(self, kpts_est, min_conf=0.3):
        with torch.no_grad():
            img_points = self.project(kpts_est)
        conf = (self.conf>min_conf)
        err = self.K[..., 0:1, 0].mean() * torch.norm(img_points - self.keypoints, dim=-1) * conf
        if len(err.shape) == 3:
            err = err.sum(dim=1)
            conf = conf.sum(dim=1)
        err = err.sum(dim=0)/(1e-5 + conf.sum(dim=0))
        return conf, err

    def check_at_start(self, kpts_est, **kwargs):
        if len(self.index_est) > 0:
            names = [str(i) for i in self.index_est]
        elif len(self.ranges_est) > 0:
            names = [str(i) for i in range(self.ranges_est[0], self.ranges_est[1])]
        else:
            names = [str(i) for i in range(self.conf.shape[-1])]
        conf, error = self.check(kpts_est)
        valid = conf.sum(dim=0).detach().cpu().numpy()
        valid = valid.tolist()
        header = ['name', 'count']
        contents = [names, valid]
        header.append('before(pix)')
        contents.append(error.detach().cpu().numpy().tolist())
        self.cache_check = (header, contents)

    def check_at_end(self, kpts_est, **kwargs):
        conf, err_after = self.check(kpts_est)
        header, contents = self.cache_check
        header.append('after(pix)')
        contents.append(err_after.detach().cpu().numpy().tolist())
        print_table(header, contents)

class DepthLoss(LossBase):
    def __init__(self, K, Rc, Tc, depth, norm, norm_info, index_est=[]):
        super().__init__()
        P = torch.bmm(K, torch.cat([Rc, Tc], dim=-1))
        self.register_buffer('P', P)
        self.index_est = index_est
        depth = BaseKeypoints.select(depth, self.index_est, [])
        self.register_buffer('depth', depth)
        self.einsum = 'fab,fnb->fna'
        self.lossfunc = make_loss(norm, norm_info)
    
    def forward(self, kpts_est, **kwargs):
        kpts_est = BaseKeypoints.select(kpts_est, self.index_est, [])
        kpts_homo = torch.ones_like(kpts_est[..., -1:])
        kpts_homo = torch.cat([kpts_est, kpts_homo], dim=-1)
        point_cam = torch.einsum(self.einsum, self.P, kpts_homo)
        depth = point_cam[..., -1]
        conf = self.depth[..., 1]
        loss = self.lossfunc(depth[..., None], self.depth[..., :1], conf)
        return loss