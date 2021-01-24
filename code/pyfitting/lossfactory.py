'''
  @ Date: 2020-11-19 17:46:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-22 16:51:55
  @ FilePath: /EasyMocap/code/pyfitting/lossfactory.py
'''
import torch
from .operation import projection, batch_rodrigues

def ReprojectionLoss(keypoints3d, keypoints2d, K, Rc, Tc, inv_bbox_sizes, norm='l2'):
    img_points = projection(keypoints3d, K, Rc, Tc)
    residual = (img_points - keypoints2d[:, :, :2]) * keypoints2d[:, :, -1:]
    # squared_res: (nFrames, nJoints, 2)
    if norm == 'l2':
        squared_res = (residual ** 2) * inv_bbox_sizes
    elif norm == 'l1':
        squared_res = torch.abs(residual) * inv_bbox_sizes
    else:
        import ipdb; ipdb.set_trace()
    return torch.sum(squared_res)

class SMPLAngleLoss:
    def __init__(self, keypoints, model_type='smpl'):
        if keypoints.shape[1] <= 15:
            use_feet = False
            use_head = False
        else:
            use_feet = keypoints[:, [19, 20, 21, 22, 23, 24], -1].sum() > 0.1
            use_head = keypoints[:, [15, 16, 17, 18], -1].sum() > 0.1
        if model_type == 'smpl':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14, 20, 21, 22, 23]
        elif model_type == 'smplh':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        elif model_type == 'smplx':
            SMPL_JOINT_ZERO_IDX = [3, 6, 9, 10, 11, 13, 14]
        else:
            raise NotImplementedError
        if not use_feet:
            SMPL_JOINT_ZERO_IDX.extend([7, 8])
        if not use_head:
            SMPL_JOINT_ZERO_IDX.extend([12, 15])
        SMPL_POSES_ZERO_IDX = [[j for j in range(3*i, 3*i+3)] for i in SMPL_JOINT_ZERO_IDX]
        SMPL_POSES_ZERO_IDX = sum(SMPL_POSES_ZERO_IDX, [])
        # SMPL_POSES_ZERO_IDX.extend([36, 37, 38, 45, 46, 47])
        self.idx = SMPL_POSES_ZERO_IDX

    def loss(self, poses):
        return torch.sum(torch.abs(poses[:, self.idx]))

def SmoothLoss(body_params, keys, weight_loss, span=4, model_type='smpl'):
    spans = [i for i in range(1, span)]
    span_weights = {i:1/i for i in range(1, span)}
    span_weights = {key: i/sum(span_weights) for key, i in span_weights.items()}
    loss_dict = {}
    nFrames = body_params['poses'].shape[0]
    nPoses = body_params['poses'].shape[1]
    if model_type == 'smplh' or model_type == 'smplx':
        nPoses = 66
    for key in ['poses', 'Th', 'poses_hand', 'expression']:
        if key not in keys:
            continue
        k = 'smooth_' + key
        if k in weight_loss.keys() and weight_loss[k] > 0.:
            loss_dict[k] = 0.
            for span in spans:
                if key == 'poses_hand':
                    val = torch.sum((body_params['poses'][span:, 66:] - body_params['poses'][:nFrames-span, 66:])**2)
                else:
                    val = torch.sum((body_params[key][span:, :nPoses] - body_params[key][:nFrames-span, :nPoses])**2)
                loss_dict[k] += span_weights[span] * val
        k = 'smooth_' + key + '_l1'
        if k in weight_loss.keys() and weight_loss[k] > 0.:
            loss_dict[k] = 0.
            for span in spans:
                if key == 'poses_hand':
                    val = torch.sum((body_params['poses'][span:, 66:] - body_params['poses'][:nFrames-span, 66:]).abs())
                else:
                    val = torch.sum((body_params[key][span:, :nPoses] - body_params[key][:nFrames-span, :nPoses]).abs())
                loss_dict[k] += span_weights[span] * val
    # smooth rotation
    rot = batch_rodrigues(body_params['Rh'])
    key, k = 'Rh', 'smooth_Rh'
    if key in keys and k in weight_loss.keys() and weight_loss[k] > 0.:
        loss_dict[k] = 0.
        for span in spans:
            val = torch.sum((rot[span:, :] - rot[:nFrames-span, :])**2)
            loss_dict[k] += span_weights[span] * val
    return loss_dict

def RegularizationLoss(body_params, body_params_init, weight_loss):
    loss_dict = {}
    for key in ['poses', 'shapes', 'Th', 'hands', 'head', 'expression']:
        if 'init_'+key in weight_loss.keys() and weight_loss['init_'+key] > 0.:
            if key == 'poses':
                loss_dict['init_'+key] = torch.sum((body_params[key][:, :66] - body_params_init[key][:, :66])**2)
            elif key == 'hands':
                loss_dict['init_'+key] = torch.sum((body_params['poses'][: , 66:66+12] - body_params_init['poses'][:, 66:66+12])**2)
            elif key == 'head':
                loss_dict['init_'+key] = torch.sum((body_params['poses'][: , 78:78+9] - body_params_init['poses'][:, 78:78+9])**2)
            elif key in body_params.keys():
                loss_dict['init_'+key] = torch.sum((body_params[key] - body_params_init[key])**2)
    for key in ['poses', 'shapes', 'hands', 'head', 'expression']:
        if 'reg_'+key in weight_loss.keys() and weight_loss['reg_'+key] > 0.:
            if key == 'poses':
                loss_dict['reg_'+key] = torch.sum((body_params[key][:, :66])**2)
            elif key == 'hands':
                loss_dict['reg_'+key] = torch.sum((body_params['poses'][: , 66:66+12])**2)
            elif key == 'head':
                loss_dict['reg_'+key] = torch.sum((body_params['poses'][: , 78:78+9])**2)
            elif key in body_params.keys():
                loss_dict['reg_'+key] = torch.sum((body_params[key])**2)
    return loss_dict