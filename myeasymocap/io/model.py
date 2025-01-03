import os
import torch
import numpy as np
from easymocap.bodymodel.smpl import SMPLModel

from easymocap.mytools.debug_utils import log

def try_to_download_SMPL(model_dir):
    cmd = 'gdown 1qIq0CBBj-O6wVc9nJXG-JDEtWPzRQ4KC'
    os.system(cmd)
    os.makedirs(model_dir, exist_ok=True)
    cmd = 'unzip pare-github-data.zip -d {}'.format(model_dir)
    print('[RUN] {}'.format(cmd))
    os.system(cmd)

class SMPLLoader:
    def __init__(self, model_path, regressor_path, return_keypoints=True):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if not os.path.exists(model_path):
            log('[SMPL] Model not found in `{}`'.format(model_path))
            log('[SMPL] Downloading model to `{}`'.format(model_path))
            try_to_download_SMPL('models/pare')
        assert os.path.exists(model_path), f'{model_path} not exists'
        if not os.path.exists(regressor_path):
            if regressor_path.endswith('J_regressor_body25.npy'):
                url = 'https://github.com/zju3dv/EasyMocap/raw/master/data/smplx/J_regressor_body25.npy'
                os.makedirs(os.path.dirname(regressor_path), exist_ok=True)
                cmd = 'wget {} -O {}'.format(url, regressor_path)
                os.system(cmd)
        assert os.path.exists(regressor_path), f'{regressor_path} not exists'
        log('[SMPL] Loading model in `{}`'.format(model_path))
        log('[SMPL] Using keypoints regressor `{}`'.format(regressor_path))
        smplmodel = SMPLModel(model_path=model_path,
                              model_type='smpl', device=device,
                              regressor_path=regressor_path,
                              NUM_SHAPES=10,
                              )
        self.smplmodel = smplmodel
        self.return_keypoints = return_keypoints

    def __call__(self,):
        return {
            'body_model': self.smplmodel, 
            'model': self.forward}
    
    def forward(self, params, ret_vertices=False):
        if ret_vertices:
            keypoints = self.smplmodel.vertices(params, return_tensor=True)
        else:
            keypoints = self.smplmodel.keypoints(params, return_tensor=True)
        ret = {
            'keypoints': keypoints
        }
        ret.update(params)
        return ret

class MANOLoader:
    def __init__(self, cfg_path, model_path, regressor_path, num_pca_comps=45, use_pca=False, use_flat_mean=False):
        log('[MANO] Loading model in `{}`'.format(model_path))
        log('[MANO] Using keypoints regressor `{}`'.format(regressor_path))
        assert os.path.exists(model_path), f'{model_path} not exists, Please download it from `mano.is.tue.mpg.de`'
        if not os.path.exists(regressor_path) and regressor_path.endswith('J_regressor_mano_LEFT.txt'):
            url = 'https://raw.githubusercontent.com/zju3dv/EasyMocap/master/data/smplx/J_regressor_mano_LEFT.txt'
            os.makedirs(os.path.dirname(regressor_path), exist_ok=True)
            cmd = 'wget {} -O {}'.format(url, regressor_path)
            os.system(cmd)
        assert os.path.exists(regressor_path), f'{regressor_path} not exists'
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        from easymocap.config import Config, load_object
        cfg_data = Config.load(cfg_path)
        cfg_data['args']['model_path'] = model_path
        cfg_data['args']['regressor_path'] = regressor_path
        cfg_data['args']['cfg_hand']['num_pca_comps'] = num_pca_comps
        cfg_data['args']['cfg_hand']['use_pca'] = use_pca
        cfg_data['args']['cfg_hand']['use_flat_mean'] = use_flat_mean
        model = load_object(cfg_data.module, cfg_data.args)
        self.manomodel = model

    def __call__(self,):
        return {
            'hand_model': self.manomodel, 
            'model': self.forward}
    
    def forward(self, params):
        keypoints = self.manomodel.keypoints(params, return_tensor=True)
        ret = {
            'keypoints': keypoints
        }
        ret.update(params)
        return ret

class MANOLoader_lr:
    def __init__(self, cfg_path, model_path, regressor_path, num_pca_comps=45, use_pca=False):
        self.Model_l = MANOLoader(cfg_path, model_path, regressor_path, num_pca_comps, use_pca)
        self.Model_r = MANOLoader(cfg_path, model_path.replace('LEFT','RIGHT'), regressor_path.replace('LEFT','RIGHT'), num_pca_comps, use_pca)
    def __call__(self,):
        ret={}
        out1 = self.Model_l()
        for key in out1.keys():
            ret[key+'_l'] = out1[key]
        out2 = self.Model_r()
        for key in out1.keys():
            ret[key+'_r'] = out2[key]
        return ret

class SMPLHLoader:
    def __init__(self, path):
        from easymocap.config import Config, load_object
        cfg_data = Config.load(path)
        self.model = load_object(cfg_data.module, cfg_data.args)
    
    def __call__(self,):
        return {
            'smplh_model': self.model, 
            'model': self.forward}
    
    def forward(self, params):
        keypoints = self.model(**params, return_verts=False, return_tensor=True)
        ret = {
            'keypoints': keypoints.clone(),#
            'keypoints_body': keypoints[...,:25,:].clone(),
            'keypoints_handlr': keypoints[...,25:,:].clone()

        }
        ret.update(params)
        return ret
