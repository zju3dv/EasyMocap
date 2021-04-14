'''
  @ Date: 2021-04-13 20:12:58
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 22:51:39
  @ FilePath: /EasyMocapRelease/easymocap/pipeline/weight.py
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
    else:
        raise NotImplementedError
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight
