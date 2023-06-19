from typing import Any
import numpy as np
from easymocap.mytools.debug_utils import mywarn, log

def solve_translation(X, x, K):
    A = np.zeros((2*X.shape[0], 3))
    b = np.zeros((2*X.shape[0], 1))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    for nj in range(X.shape[0]):
        A[2*nj, 0] = 1
        A[2*nj + 1, 1] = 1
        A[2*nj, 2] = -(x[nj, 0] - cx)/fx
        A[2*nj+1, 2] = -(x[nj, 1] - cy)/fy
        b[2*nj, 0] = X[nj, 2]*(x[nj, 0] - cx)/fx - X[nj, 0]
        b[2*nj+1, 0] = X[nj, 2]*(x[nj, 1] - cy)/fy - X[nj, 1]
        A[2*nj:2*nj+2, :] *= x[nj, 2]
        b[2*nj:2*nj+2, :] *= x[nj, 2]
    trans = np.linalg.inv(A.T @ A) @ A.T @ b
    return trans.T[0]

class MeanShapes:
    def __init__(self, keys, dim=0) -> None:
        self.keys = keys
        self.dim = dim
    
    def __call__(self, params):
        for key in self.keys:
            log('[{}] Mean {}: {}'.format(self.__class__.__name__, key, params[key].shape))
            params[key] = params[key].mean(axis=self.dim, keepdims=True)
            log('[{}] Mean {}: {}'.format(self.__class__.__name__, key, params[key].shape))

class InitTranslation:
    def __init__(self, solve_T=True, solve_R=False) -> None:
        self.solve_T = solve_T
        self.solve_R = solve_R
    
    def __call__(self, body_model, params, cameras, keypoints):
        nJoints = 15 # 只使用主要的15个点
        params['Th'] = np.zeros_like(params['Th'])
        kpts1 = body_model.keypoints(params, return_tensor=False)
        for i in range(kpts1.shape[0]):
            k2d = keypoints[i, :nJoints]
            if k2d[:, -1].sum() < nJoints / 2:
                mywarn('[{}] No valid keypoints in frame {}'.format(self.__class__.__name__, i))
                params['Th'][i] = params['Th'][i-1]
                continue
            trans = solve_translation(kpts1[i, :nJoints], k2d, cameras['K'][i])
            params['Th'][i] += trans
        # params['shapes'] = params['shapes'].mean(0, keepdims=True)
        return {'params': params}

class InitParams:
    def __init__(self, num_poses=69, num_shapes=10, rootid=8, share_shape=True, init_trans=0.) -> None:
        self.num_poses = num_poses
        self.num_shapes = num_shapes
        self.rootid = rootid
        self.share_shape = share_shape
        self.init_trans = init_trans

    def __call__(self, **kwargs):
        """
            keypoints3d: (nFrames, nJoints, 4) or (nFrames, nPerson, nFrames, 4)
        """
        key = list(kwargs.keys())[0]
        keypoints3d = kwargs[key]
        if keypoints3d.ndim == 4:
            shape = (keypoints3d.shape[:2])
        elif keypoints3d.ndim == 3:
            shape = (keypoints3d.shape[0],)
        else:
            raise ValueError('keypoints3d must be 3 or 4 dim')
        params={
            'Rh': np.zeros((*shape, 3),dtype=np.float32),
            'Th': np.zeros((*shape, 3),dtype=np.float32),
            'poses': np.zeros((*shape, self.num_poses),dtype=np.float32),
            'shapes': np.zeros((*shape, self.num_shapes),dtype=np.float32)
        }
        # TODO: check the root confidence and interpolate
        # 初始化
        if key == 'keypoints3d':
            params['Th'] = keypoints3d[..., self.rootid, :3]
        else:
            mywarn('[{}] Not used keypoints3d, set to {}'.format(self.__class__.__name__, self.init_trans))
            params['Th'][:, 2] = self.init_trans
        if self.share_shape:
            params['shapes'] = params['shapes'].mean(0, keepdims=True)
        return {'params': params}

class Init_params_and_target_poses(InitParams):
    def __call__(self, params_smplh, model):
        """
            keypoints3d: (nFrames, nJoints, 4) or (nFrames, nPerson, nFrames, 4)
        """
        out = model(params_smplh)
        keypoints3d = out['keypoints'].cpu().detach().numpy()
        ret = super().__call__(keypoints3d)
        for key in params_smplh.keys():
            ret['params'][key] = params_smplh[key]
            ret['target_'+key] = params_smplh[key]
        return ret
