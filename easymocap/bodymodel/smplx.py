import torch
import torch.nn as nn
from .base import Model
from .smpl import SMPLModel, SMPLLayerEmbedding, read_pickle, to_tensor
from os.path import join
import numpy as np

def read_hand(path, use_pca, use_flat_mean, num_pca_comps):
    data = read_pickle(path)
    mean = data['hands_mean'].reshape(1, -1).astype(np.float32)
    mean_full = mean
    components_full = data['hands_components'].astype(np.float32)
    weight = np.diag(components_full @ components_full.T)
    components = components_full[:num_pca_comps]
    weight = weight[:num_pca_comps]
    if use_flat_mean:
        mean = np.zeros_like(mean)
    return mean, components, weight, mean_full, components_full

class MANO(SMPLModel):
    def __init__(self, cfg_hand, **kwargs):
        super().__init__(**kwargs)
        self.name = 'mano'
        self.use_root_rot = False
        mean, components, weight, mean_full, components_full = read_hand(kwargs['model_path'], **cfg_hand)
        self.register_buffer('mean', to_tensor(mean, dtype=self.dtype))
        self.register_buffer('components', to_tensor(components, dtype=self.dtype))
        self.cfg_hand = cfg_hand
        self.to(self.device)
        if cfg_hand.use_pca:
            self.NUM_POSES = cfg_hand.num_pca_comps
    
    def extend_poses(self, poses, **kwargs):
        if poses.shape[-1] == self.mean.shape[-1] + 3:
            return poses
        if self.cfg_hand.use_pca:
            poses = poses @ self.components
        if kwargs.get('pose2rot', True):
            poses = super().extend_poses(poses+self.mean, **kwargs)
        else:
            poses = super().extend_poses(poses, **kwargs)
        return poses
    
    def jacobian_posesfull_poses(self, poses, poses_full):
        if self.cfg_hand.use_pca:
            jacobian = self.components.t()
            zero_root = torch.zeros((3, poses.shape[-1]), dtype=poses.dtype, device=poses.device)
            jacobian = torch.cat([zero_root, jacobian], dim=0)
        else:
            jacobian = super().jacobian_posesfull_poses(poses, poses_full)
        return jacobian
class MANOLR(Model):
    def __init__(self, model_path, regressor_path, cfg_hand, **kwargs):
        super().__init__()
        self.name = 'manolr'
        keys = list(model_path.keys())
        # stack 方式：(nframes, nhand x ndim)
        self.keys = keys
        modules_hand = {}
        faces = []
        v_template = []
        cnt = 0
        for key in keys:
            modules_hand[key] = MANO(cfg_hand, model_path=model_path[key], regressor_path=regressor_path[key], **kwargs)
            v_template.append(modules_hand[key].v_template.cpu().numpy())
            faces.append(modules_hand[key].faces + cnt)
            cnt += v_template[-1].shape[0]
            self.device = modules_hand[key].device
            self.dtype = modules_hand[key].dtype
            if key == 'right':
                modules_hand[key].shapedirs[:, 0] *= -1
                modules_hand[key].j_shapedirs[:, 0] *= -1
        self.faces = np.vstack(faces)
        self.v_template = np.vstack(v_template)
        self.modules_hand = nn.ModuleDict(modules_hand)
        self.to(self.device)

    def init_params(self, **kwargs):
        param_all = {}
        for key in self.keys:
            param = self.modules_hand[key].init_params(**kwargs)
            param_all[key] = param
        if False:
            params = {k: torch.cat([param_all[key][k] for key in self.keys], dim=-1) for k in param.keys()}
        else:
            params = {k: np.concatenate([param_all[key][k] for key in self.keys], axis=-1) for k in param.keys() if k != 'shapes'}
            params['shapes'] = param_all['left']['shapes']
        return params

    def split(self, params):
        params_split = {}
        for imodel, model in enumerate(self.keys):
            param_= params.copy()
            for key in ['poses', 'shapes', 'Rh', 'Th']:
                if key not in params.keys():continue
                if key == 'shapes':
                    continue
                shape = params[key].shape[-1]
                start = shape//len(self.keys)*imodel
                end = shape//len(self.keys)*(imodel+1)
                param_[key] = params[key][:, start:end]
            params_split[model] = param_
        return params_split

    def forward(self, **params):
        params_split = self.split(params)
        rets = []
        for imodel, model in enumerate(self.keys):
            ret = self.modules_hand[model](**params_split[model])
            rets.append(ret)
        if params.get('return_tensor', True):
            rets = torch.cat(rets, dim=1)
        else:
            rets = np.concatenate(rets, axis=1)
        return rets

    def extend_poses(self, poses, **kwargs):
        params_split = self.split({'poses': poses})
        rets = []
        for imodel, model in enumerate(self.keys):
            poses = params_split[model]['poses']
            poses = self.modules_hand[model].extend_poses(poses)
            rets.append(poses)
        poses = torch.cat(rets, dim=1)
        return poses

    def export_full_poses(self, poses, **kwargs):
        params_split = self.split({'poses': poses})
        rets = []
        for imodel, model in enumerate(self.keys):
            poses = torch.Tensor(params_split[model]['poses']).to(self.device)
            poses = self.modules_hand[model].extend_poses(poses)
            rets.append(poses)
        poses = torch.cat(rets, dim=1)
        return poses.detach().cpu().numpy()

class SMPLHModel(SMPLModel):
    def __init__(self, mano_path, cfg_hand, **kwargs):
        super().__init__(**kwargs)
        self.NUM_POSES = self.NUM_POSES - 90
        meanl, componentsl, weight_l, self.mean_full_l, self.components_full_l = read_hand(join(mano_path, 'MANO_LEFT.pkl'), **cfg_hand)
        meanr, componentsr, weight_r, self.mean_full_r, self.components_full_r = read_hand(join(mano_path, 'MANO_RIGHT.pkl'), **cfg_hand)
        self.register_buffer('weight_l', to_tensor(weight_l, dtype=self.dtype))
        self.register_buffer('weight_r', to_tensor(weight_r, dtype=self.dtype))
        self.register_buffer('meanl', to_tensor(meanl, dtype=self.dtype))
        self.register_buffer('meanr', to_tensor(meanr, dtype=self.dtype))
        self.register_buffer('componentsl', to_tensor(componentsl, dtype=self.dtype))
        self.register_buffer('componentsr', to_tensor(componentsr, dtype=self.dtype))

        self.register_buffer('jacobian_posesfull_poses_', self._jacobian_posesfull_poses())
        self.NUM_HANDS = cfg_hand.num_pca_comps if cfg_hand.use_pca else 45
        self.cfg_hand = cfg_hand
        self.to(self.device)
    
    def _jacobian_posesfull_poses(self):
        # TODO: cache this 
        # | body_full/body | 0 | 0 |
        # |      0         | l | 0 |
        # |      0         | 0 | r |
        eye_right = torch.eye(self.NUM_POSES, dtype=self.dtype)
        # 
        jac_handl = self.componentsl.t()
        jac_handr = self.componentsr.t()
        output = torch.zeros((self.NUM_POSES_FULL, self.NUM_POSES+jac_handl.shape[1]*2), dtype=self.dtype)
        if self.use_root_rot:
            raise NotImplementedError
        else:
            output[3:3+self.NUM_POSES, :self.NUM_POSES] = eye_right
            output[3+self.NUM_POSES:3+self.NUM_POSES+jac_handl.shape[0], \
                self.NUM_POSES:self.NUM_POSES+jac_handl.shape[1]] = jac_handl
            output[3+self.NUM_POSES+jac_handl.shape[0]:3+self.NUM_POSES+2*jac_handl.shape[0], \
                self.NUM_POSES+jac_handl.shape[1]:self.NUM_POSES+jac_handl.shape[1]*2] = jac_handr            
        return output

    def init_params(self, nFrames=1, nShapes=1, nPerson=1, ret_tensor=False, add_scale=False):
        params = super().init_params(nFrames, nShapes, nPerson, ret_tensor, add_scale=add_scale)
        handl = np.zeros((nFrames, self.NUM_HANDS))
        handr = np.zeros((nFrames, self.NUM_HANDS))
        if nPerson > 1:
            handl = handl[:, None].repeat(nPerson, axis=1)
            handr = handr[:, None].repeat(nPerson, axis=1)
        if ret_tensor:
            handl = to_tensor(handl, self.dtype, self.device)
            handr = to_tensor(handr, self.dtype, self.device)
        params['handl'] = handl
        params['handr'] = handr
        return params
    
    def extend_poses(self, poses, handl=None, handr=None, **kwargs):
        if poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        poses = super().extend_poses(poses)
        if handl is None:
            handl = self.meanl.clone()
            handr = self.meanr.clone()
            handl = handl.expand(poses.shape[0], -1)
            handr = handr.expand(poses.shape[0], -1)
        else:
            if self.cfg_hand.use_pca:
                handl = handl @ self.componentsl
                handr = handr @ self.componentsr
            handl = handl +self.meanl
            handr = handr +self.meanr
        poses = torch.cat([poses, handl, handr], dim=-1)
        return poses
    
    def export_full_poses(self, poses, handl, handr, **kwargs):
        poses = torch.Tensor(poses).to(self.device)
        handl = torch.Tensor(handl).to(self.device)
        handr = torch.Tensor(handr).to(self.device)
        poses = self.extend_poses(poses, handl, handr)
        return poses.detach().cpu().numpy()

class SMPLHModelEmbedding(SMPLHModel):
    def __init__(self, vposer_ckpt='data/body_models/vposer_v02', **kwargs):
        super().__init__(**kwargs)
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        vposer, _ = load_model(vposer_ckpt, 
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True)
        vposer.to(self.device)
        self.vposer = vposer
        self.vposer_dim = 32
        self.NUM_POSES = self.vposer_dim
    
    def decode(self, poses, add_rot=True):
        if poses.shape[-1] == 66 and add_rot:
            return poses
        elif poses.shape[-1] == 63 and not add_rot:
            return poses
        assert poses.shape[-1] == self.vposer_dim, poses.shape
        ret = self.vposer.decode(poses)
        poses_body = ret['pose_body'].reshape(poses.shape[0], -1)
        if add_rot:
            zero_rot = torch.zeros((poses.shape[0], 3), dtype=poses.dtype, device=poses.device)
            poses_body = torch.cat([zero_rot, poses_body], dim=-1)
        return poses_body

    def extend_poses(self, poses, handl, handr, **kwargs):
        if poses.shape[-1] == self.NUM_POSES_FULL:
            return poses
        zero_rot = torch.zeros((poses.shape[0], 3), dtype=poses.dtype, device=poses.device)
        poses_body = self.decode(poses, add_rot=False)
        if self.cfg_hand.use_pca:
            handl = handl @ self.componentsl
            handr = handr @ self.componentsr
        handl = handl +self.meanl
        handr = handr +self.meanr
        poses = torch.cat([zero_rot, poses_body, handl, handr], dim=-1)
        return poses
    
    def export_full_poses(self, poses, handl, handr, **kwargs):
        poses = torch.Tensor(poses).to(self.device)
        handl = torch.Tensor(handl).to(self.device)
        handr = torch.Tensor(handr).to(self.device)
        poses = self.extend_poses(poses, handl, handr)
        return poses.detach().cpu().numpy()
