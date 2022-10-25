import numpy as np
import cv2
from ..dataset.config import CONFIG
from ..config import load_object
from ..mytools.debug_utils import log, mywarn, myerror
import torch
from tqdm import tqdm, trange

def svd_rot(src, tgt, reflection=False, debug=False):
    # optimum rotation matrix of Y
    A = np.matmul(src.transpose(0, 2, 1), tgt)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.transpose(0, 2, 1)
    T = np.matmul(V, U.transpose(0, 2, 1))
    # does the current solution use a reflection?
    have_reflection = np.linalg.det(T) < 0

    # if that's not what was specified, force another reflection
    V[have_reflection, :, -1] *= -1
    s[have_reflection, -1] *= -1
    T = np.matmul(V, U.transpose(0, 2, 1))
    if debug:
        err = np.linalg.norm(tgt - src @ T.T, axis=1)
        print('[svd] ', err)
    return T

def batch_invRodrigues(rot):
    res = []
    for r in rot:
        v = cv2.Rodrigues(r)[0]
        res.append(v)
    res = np.stack(res)
    return res[:, :, 0]

class BaseInit:
    def __init__(self) -> None:
        pass

    def __call__(self, body_model, body_params, infos):\
        return body_params

class Remove(BaseInit):
    def __init__(self, key, index) -> None:
        super().__init__()
        self.key = key
        self.index = index
    
    def __call__(self, body_model, body_params, infos):
        infos[self.key][..., self.index, :] = 0
        return super().__call__(body_model, body_params, infos)

class CheckKeypoints:
    def __init__(self, type) -> None:
        # this class is used to check if the provided keypoints3d
        self.type = type
        self.body_config = CONFIG[type]
        self.hand_config = CONFIG['hand']
    
    def __call__(self, body_model, body_params, infos):
        for key in ['keypoints3d', 'handl3d', 'handr3d']:
            if key not in infos.keys(): continue
            keypoints = infos[key]
            conf = keypoints[..., -1]
            keypoints[conf<0.1] = 0
            if key == 'keypoints3d':
                continue
                import ipdb;ipdb.set_trace()
                # limb_length = np.linalg.norm(keypoints[:, , :3], axis=2)
        return body_params

class InitRT:
    def __init__(self, torso) -> None:
        self.torso = torso

    def __call__(self, body_model, body_params, infos):
        keypoints3d = infos['keypoints3d']
        if torch.is_tensor(keypoints3d):
            keypoints3d = keypoints3d.detach().cpu().numpy()
        temp_joints = body_model.keypoints(body_params, return_tensor=False)
        
        torso = keypoints3d[..., self.torso, :3].copy()
        torso_temp = temp_joints[..., self.torso, :3].copy()
        # here use the first id of torso as the rotation center
        root, root_temp = torso[..., :1, :], torso_temp[..., :1, :]
        torso = torso - root
        torso_temp = torso_temp - root_temp
        conf = (keypoints3d[..., self.torso, 3] > 0.).all(axis=-1)
        if not conf.all():
            myerror("The torso in frames {} is not valid, please check the 3d keypoints".format(np.where(~conf)))
        if len(torso.shape) == 3:
            R = svd_rot(torso_temp, torso)
            R_flat = R
            T = np.matmul(- root_temp, R.transpose(0, 2, 1)) + root
        else:
            R_flat = svd_rot(torso_temp.reshape(-1, *torso_temp.shape[-2:]), torso.reshape(-1, *torso.shape[-2:]))
            R = R_flat.reshape(*torso.shape[:2], 3, 3)
            T = np.matmul(- root_temp, R.swapaxes(-1, -2)) + root
        for nf in np.where(~conf)[0]:
            # copy previous frames
            mywarn('copy {} from {}'.format(nf, nf-1))
            R[nf] = R[nf-1]
            T[nf] = T[nf-1]
        body_params['Th'] = T[..., 0, :]
        rvec = batch_invRodrigues(R_flat)
        if len(torso.shape) > 3:
            rvec = rvec.reshape(*torso.shape[:2], 3)
        body_params['Rh'] = rvec
        return body_params

    def __str__(self) -> str:
        return "[Initialize] svd with torso: {}".format(self.torso)

class TriangulatorWrapper:
    def __init__(self, module, args):
        self.triangulator = load_object(module, args)

    def __call__(self, body_model, body_params, infos):
        infos['RT'] = torch.cat([infos['Rc'], infos['Tc']], dim=-1)
        data = {
            'RT': infos['RT'].numpy(),
        }
        for key in self.triangulator.keys:
            if key not in infos.keys():
                continue
            data[key] = infos[key].numpy()
            data[key+'_unproj'] = infos[key+'_unproj'].numpy()
            data[key+'_distort'] = infos[key+'_distort'].numpy()            
        results = self.triangulator(data)[0]
        for key, val in results.items():
            if key == 'id': continue
            infos[key] = torch.Tensor(val[None].astype(np.float32))
        body_params = body_model.init_params(nFrames=1, add_scale=True)
        return body_params

class CheckRT:
    def __init__(self, T_thres, window):
        self.T_thres = T_thres
        self.window = window

    def __call__(self, body_model, body_params, infos):
        Th = body_params['Th']
        if len(Th.shape) == 3:
            for nper in range(Th.shape[1]):
                for nf in trange(1, Th.shape[0], desc='Check Th of {}'.format(nper)):
                    if nf > self.window:
                        tpre = Th[nf-self.window:nf, nper]
                    else:
                        tpre = Th[:nf, nper]
                    tpre = tpre.mean(axis=0)
                    tnow = Th[nf  , nper]
                    dist = np.linalg.norm(tnow - tpre)
                    if dist > self.T_thres:
                        mywarn('[Check Th] distance in frame {} = {} larger than {}'.format(nf, dist, self.T_thres))
                        Th[nf, nper] = tpre
        body_params['Th'] = Th
        return body_params

class Scale:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, body_model, body_params, infos):
        scale = body_params.pop('scale')[0, 0]
        if scale < 1.1 and scale > 0.9:
            return body_params
        print('scale = ', scale)
        for key in self.keys:
            if key not in infos.keys():
                continue
            infos[key] /= scale
        infos['Tc'] /= scale
        infos['RT'][..., -1] *= scale
        infos['scale'] = scale
        return body_params
