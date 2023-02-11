'''
  @ Date: 2022-08-12 20:34:15
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-18 14:47:23
  @ FilePath: /EasyMocapPublic/easymocap/multistage/base_ops.py
'''
import torch

class BeforeAfterBase:
    def __init__(self, model) -> None:
        pass

    def start(self, body_params):
        # operation before the optimization
        return body_params
    
    def before(self, body_params):
        # operation in each optimization step
        return body_params
    
    def final(self, body_params):
        # operation after the optimization
        return body_params

class SkipPoses(BeforeAfterBase):
    def __init__(self, index, nPoses) -> None:
        self.index = index
        self.nPoses = nPoses
        self.copy_index = [i for i in range(nPoses) if i not in index]

    def before(self, body_params):
        poses = body_params['poses']
        poses_copy = torch.zeros_like(poses)
        # print(poses.shape)
        poses_copy[..., self.copy_index] = poses[..., self.copy_index]
        body_params['poses'] = poses_copy
        return body_params