'''
  @ Date: 2020-11-20 13:34:54
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 20:09:40
  @ FilePath: /EasyMocap/code/smplmodel/body_param.py
'''
import numpy as np

def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th']:
        output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output

def select_nf(params_all, nf):
    output = {}
    for key in ['poses', 'Rh', 'Th']:
        output[key] = params_all[key][nf:nf+1, :]
    if params_all['shapes'].shape[0] == 1:
        output['shapes'] = params_all['shapes']
    else:
        output['shapes'] = params_all['shapes'][nf:nf+1, :]
    return output

def init_params(nFrames=1):
    params = {
        'poses': np.zeros((nFrames, 72)),
        'shapes': np.zeros((1, 10)),
        'Rh': np.zeros((nFrames, 3)),
        'Th': np.zeros((nFrames, 3)),
    }
    return params

class Config:
    OPT_R = False
    OPT_T = False
    OPT_POSE = False
    OPT_SHAPE = False
    VERBOSE = False