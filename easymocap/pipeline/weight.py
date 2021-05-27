'''
  @ Date: 2021-04-13 20:12:58
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-27 17:04:47
  @ FilePath: /EasyMocap/easymocap/pipeline/weight.py
'''
def load_weight_shape(opts):
    weight = {'s3d': 1., 'reg_shapes': 5e-3}
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight

def load_weight_pose(model, opts):
    if model == 'smpl':
        weight = {
            'k3d': 1., 'reg_poses_zero': 1e-2, 'smooth_body': 5e-1,
            'smooth_poses': 1e-1, 'reg_poses': 1e-3,
            'k2d': 1e-4
        }
    elif model == 'smplh':
        weight = {
            'k3d': 1., 'k3d_hand': 5.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1, 'smooth_hand': 1e-3,
            'reg_hand': 1e-4,
            'k2d': 1e-4
        }
    elif model == 'smplx':
        weight = {
            'k3d': 1., 'k3d_hand': 5., 'k3d_face': 2.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1, 'smooth_hand': 1e-3,
            'reg_hand': 1e-4, 'reg_expr': 1e-2, 'reg_head': 1e-2,
            'k2d': 1e-4
        }
    elif model == 'mano':
        weight = {
            'k3d': 1e2, 'k2d': 1e-3,
            'reg_poses': 1e-3, 'smooth_body': 1e2
        }
    else:
        print(model)
        raise NotImplementedError
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight

def load_weight_pose2d(model, opts):
    if model == 'smpl':
        weight = {
            'k2d': 2e-4, 
            'init_poses': 1e-3, 'init_shapes': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1,
        }
    elif model == 'smplh':
        raise NotImplementedError
    elif model == 'smplx':
        raise NotImplementedError
    else:
        weight = {}
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight