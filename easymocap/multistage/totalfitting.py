'''
  @ Date: 2022-07-28 14:39:23
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-12 21:42:12
  @ FilePath: /EasyMocapPublic/easymocap/multistage/totalfitting.py
'''
import torch

from ..bodymodel.lbs import batch_rodrigues
from .torchgeometry import rotation_matrix_to_axis_angle, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, quaternion_to_axis_angle
import numpy as np
from .base_ops import BeforeAfterBase

def compute_twist_rotation(rotation_matrix, twist_axis):
    '''
    Compute the twist component of given rotation and twist axis
    https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
    Parameters
    ----------
    rotation_matrix : Tensor (B, 3, 3,)
        The rotation to convert
    twist_axis : Tensor (B, 3,)
        The twist axis
    Returns
    -------
    Tensor (B, 3, 3)
        The twist rotation
    '''
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    twist_axis = twist_axis / (torch.norm(twist_axis, dim=1, keepdim=True) + 1e-9)

    projection = torch.einsum('bi,bi->b', twist_axis, quaternion[:, 1:]).unsqueeze(-1) * twist_axis

    twist_quaternion = torch.cat([quaternion[:, 0:1], projection], dim=1)
    twist_quaternion = twist_quaternion / (torch.norm(twist_quaternion, dim=1, keepdim=True) + 1e-9)

    twist_rotation = quaternion_to_rotation_matrix(twist_quaternion)

    twist_aa = quaternion_to_axis_angle(twist_quaternion)

    twist_angle = torch.sum(twist_aa, dim=1, keepdim=True) / torch.sum(twist_axis, dim=1, keepdim=True)

    return twist_rotation, twist_angle

class ClearTwist(BeforeAfterBase):
    def start(self, body_params):
        idx_elbow = [18-1, 19-1]
        for idx in idx_elbow:
            # x
            body_params['poses'][:, 3*idx] = 0.
            # z
            body_params['poses'][:, 3*idx+2] = 0.
        idx_wrist = [20-1, 21-1]
        for idx in idx_wrist:
            body_params['poses'][:, 3*idx:3*idx+3] = 0.
        return body_params

class SolveTwist(BeforeAfterBase):
    def __init__(self, body_model=None) -> None:
      self.body_model = body_model

    def final(self, body_params):
        T_joints, T_vertices = self.body_model.transform(body_params)
        # This transform don't consider RT
        R = batch_rodrigues(body_params['Rh'])
        template = self.body_model.keypoints({'shapes': body_params['shapes'],
            'poses': torch.zeros_like(body_params['poses'])},
            only_shape=True, return_smpl_joints=True)
        config = {
            'left': {
                'index_smpl': 20,
                'index_elbow_smpl': 18,
                'R_global': 'R_handl3d',
                'axis': torch.Tensor([[1., 0., 0.]]).to(device=T_joints.device),
            },
            'right': {
                'index_smpl': 21,
                'index_elbow_smpl': 19,
                'R_global': 'R_handr3d',
                'axis': torch.Tensor([[-1., 0., 0.]]).to(device=T_joints.device),
            }
        }
        for key in ['left', 'right']:
            cfg = config[key]
            R_wrist_add = batch_rodrigues(body_params[cfg['R_global']])
            idx_elbow = cfg['index_elbow_smpl']
            idx_wrist = cfg['index_smpl']
            pred_parent_elbow = R @ T_joints[..., idx_elbow, :3, :3]
            pred_parent_wrist = R @ T_joints[..., idx_wrist, :3, :3]
            pred_global_wrist = torch.bmm(R_wrist_add, pred_parent_wrist)
            pred_local_wrist = torch.bmm(pred_parent_wrist.transpose(-1, -2), pred_global_wrist)
            axis = rotation_matrix_to_axis_angle(pred_local_wrist)
            body_params['poses'][..., 3*idx_wrist-3:3*idx_wrist] = axis
        return body_params