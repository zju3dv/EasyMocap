'''
  @ Date: 2020-11-20 13:34:54
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-24 18:39:45
  @ FilePath: /EasyMocapRelease/code/smplmodel/body_param.py
'''
import numpy as np

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

NUM_POSES = {'smpl': 72, 'smplh': 78, 'smplx': 66 + 12 + 9}
NUM_EXPR = 10

def init_params(nFrames=1, model_type='smpl'):
    params = {
        'poses': np.zeros((nFrames, NUM_POSES[model_type])),
        'shapes': np.zeros((1, 10)),
        'Rh': np.zeros((nFrames, 3)),
        'Th': np.zeros((nFrames, 3)),
    }
    if model_type == 'smplx':
        params['expression'] = np.zeros((nFrames, NUM_EXPR))
    return params

def check_params(body_params, model_type):
    nFrames = body_params['poses'].shape[0]
    if body_params['poses'].shape[1] != NUM_POSES[model_type]:
        body_params['poses'] = np.hstack((body_params['poses'], np.zeros((nFrames, NUM_POSES[model_type] - body_params['poses'].shape[1]))))
    if model_type == 'smplx' and 'expression' not in body_params.keys():
        body_params['expression'] = np.zeros((nFrames, NUM_EXPR))
    return body_params

class Config:
    OPT_R = False
    OPT_T = False
    OPT_POSE = False
    OPT_SHAPE = False
    OPT_HAND = False
    OPT_EXPR = False
    VERBOSE = False
    MODEL = 'smpl'

def load_model(gender='neutral', use_cuda=True, model_type='smpl'):
    # prepare SMPL model
    import torch
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    from .body_model import SMPLlayer
    if model_type == 'smpl':
        body_model = SMPLlayer('data/smplx/smpl', gender=gender, device=device,
            regressor_path='data/smplx/J_regressor_body25.npy')
    elif model_type == 'smplh':
        body_model = SMPLlayer('data/smplx/smplh/SMPLH_MALE.pkl', model_type='smplh', gender=gender, device=device,
            regressor_path='data/smplx/J_regressor_body25_smplh.txt')
    elif model_type == 'smplx':
        body_model = SMPLlayer('data/smplx/smplx/SMPLX_{}.pkl'.format(gender.upper()), model_type='smplx', gender=gender, device=device,
            regressor_path='data/smplx/J_regressor_body25_smplx.txt')
    else:
        body_model = None
    body_model.to(device)
    return body_model

def check_keypoints(keypoints2d, WEIGHT_DEBUFF=1.2):
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
    MIN_CONF = 0.3
    conf = keypoints2d[..., -1]
    conf[conf<MIN_CONF] = 0
    return keypoints2d