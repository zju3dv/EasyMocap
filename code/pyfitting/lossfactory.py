'''
  @ Date: 2020-11-19 17:46:04
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 15:02:39
  @ FilePath: /EasyMocap/code/pyfitting/lossfactory.py
'''
import torch
from .operation import projection, batch_rodrigues

def ReprojectionLoss(keypoints3d, keypoints2d, K, Rc, Tc, inv_bbox_sizes):
    img_points = projection(keypoints3d, K, Rc, Tc)
    residual = (img_points - keypoints2d[:, :, :2]) * keypoints2d[:, :, 2:3]
    squared_res = (residual ** 2) * inv_bbox_sizes
    return torch.sum(squared_res)

class SMPLAngleLoss:
    def __init__(self, keypoints):
        use_feet = keypoints[:, [19, 20, 21, 22, 23, 24], -1].sum() > 0.1
        use_head = keypoints[:, [15, 16, 17, 18], -1].sum() > 0.1
        SMPL_JOINT_ZERO_IDX = [3, 6, 9, 13, 14, 20, 21, 22, 23]
        if not use_feet:
            SMPL_JOINT_ZERO_IDX.extend([7, 8])
        if not use_head:
            SMPL_JOINT_ZERO_IDX.extend([12, 15])
        SMPL_POSES_ZERO_IDX = [[j for j in range(3*i, 3*i+3)] for i in SMPL_JOINT_ZERO_IDX]
        SMPL_POSES_ZERO_IDX = sum(SMPL_POSES_ZERO_IDX, [])
        self.idx = SMPL_POSES_ZERO_IDX

    def loss(self, poses):
        return torch.sum(torch.abs(poses[:, self.idx]))

def SmoothLoss(body_params, keys, weight_loss, span=4):
    spans = [i for i in range(1, span)]
    span_weights = {i:1/i for i in range(1, span)}
    span_weights = {key: i/sum(span_weights) for key, i in span_weights.items()}
    loss_dict = {}
    nFrames = body_params['poses'].shape[0]
    for key in ['poses', 'Th']:
        k = 'smooth_' + key
        if k in weight_loss.keys() and weight_loss[k] > 0.:
            loss_dict[k] = 0.
            for span in spans:
                val = torch.sum((body_params[key][span:, :] - body_params[key][:nFrames-span, :])**2)
                loss_dict[k] += span_weights[span] * val
    # smooth rotation
    rot = batch_rodrigues(body_params['Rh'])
    key, k = 'Rh', 'smooth_Rh'
    if k in weight_loss.keys() and weight_loss[k] > 0.:
        loss_dict[k] = 0.
        for span in spans:
            val = torch.sum((rot[span:, :] - rot[:nFrames-span, :])**2)
            loss_dict[k] += span_weights[span] * val
    return loss_dict

def RegularizationLoss(body_params, body_params_init, weight_loss):
    loss_dict = {}
    for key in ['poses', 'shapes', 'Th']:
            if 'init_'+key in weight_loss.keys() and weight_loss['init_'+key] > 0.:
                loss_dict['init_'+key] = torch.sum((body_params[key] - body_params_init[key])**2)
    for key in ['poses', 'shapes']:
        if 'reg_'+key in weight_loss.keys() and weight_loss['reg_'+key] > 0.:
            loss_dict['reg_'+key] = torch.sum((body_params[key])**2)
    return loss_dict