'''
  @ Date: 2020-11-20 13:34:54
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-25 19:21:12
  @ FilePath: /EasyMocap/easymocap/smplmodel/body_param.py
'''
import numpy as np
from os.path import join

def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output

def select_nf(params_all, nf):
    output = {}
    for key in ['poses', 'Rh', 'Th']:
        output[key] = params_all[key][nf:nf+1, :]
    if 'expression' in params_all.keys():
        output['expression'] = params_all['expression'][nf:nf+1, :]
    if params_all['shapes'].shape[0] == 1:
        output['shapes'] = params_all['shapes']
    else:
        output['shapes'] = params_all['shapes'][nf:nf+1, :]
    return output

def load_model(gender='neutral', use_cuda=True, model_type='smpl', skel_type='body25', device=None, model_path='data/smplx'):
    # prepare SMPL model
    # print('[Load model {}/{}]'.format(model_type, gender))
    import torch
    if device is None:
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    from .body_model import SMPLlayer
    if model_type == 'smpl':
        if skel_type == 'body25':
            reg_path = join(model_path, 'J_regressor_body25.npy')
        elif skel_type == 'h36m':
            reg_path = join(model_path, 'J_regressor_h36m.npy')
        else:
            raise NotImplementedError
        body_model = SMPLlayer(join(model_path, 'smpl'), gender=gender, device=device,
            regressor_path=reg_path)
    elif model_type == 'smplh':
        body_model = SMPLlayer(join(model_path, 'smplh/SMPLH_MALE.pkl'), model_type='smplh', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplh.txt'))
    elif model_type == 'smplx':
        body_model = SMPLlayer(join(model_path, 'smplx/SMPLX_{}.pkl'.format(gender.upper())), model_type='smplx', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplx.txt'))
    elif model_type == 'manol' or model_type == 'manor':
        lr = {'manol': 'LEFT', 'manor': 'RIGHT'}
        body_model = SMPLlayer(join(model_path, 'smplh/MANO_{}.pkl'.format(lr[model_type])), model_type='mano', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_mano_{}.txt'.format(lr[model_type])))
    else:
        body_model = None
    body_model.to(device)
    return body_model

def check_keypoints(keypoints2d, WEIGHT_DEBUFF=1, min_conf=0.3):
    # keypoints2d: nFrames, nJoints, 3
    # 
    # wrong feet
    # if keypoints2d.shape[-2] > 25 + 42:
    #     keypoints2d[..., 0, 2] = 0
    # keypoints2d[..., [15, 16, 17, 18], -1] = 0
    # keypoints2d[..., [19, 20, 21, 22, 23, 24], -1] /= 2
    if keypoints2d.shape[-2] > 25:
        # set the hand keypoints
        keypoints2d[..., 25, :] = keypoints2d[..., 7, :]
        keypoints2d[..., 46, :] = keypoints2d[..., 4, :]
        keypoints2d[..., 25:, -1] *= WEIGHT_DEBUFF
    # reduce the confidence of hand and face
    MIN_CONF = min_conf
    conf = keypoints2d[..., -1]
    conf[conf<MIN_CONF] = 0
    return keypoints2d