'''
  @ Date: 2022-04-02 13:59:50
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-07-13 16:34:21
  @ FilePath: /EasyMocapPublic/easymocap/multistage/init_pose.py
'''
import os
import numpy as np
import cv2
from tqdm import tqdm
from os.path import join
import torch
from ..bodymodel.base import Params
from ..estimator.wrapper_base import bbox_from_keypoints
from ..mytools.writer import write_smpl
from ..mytools.reader import read_smpl

class SmoothPoses:
    def __init__(self, window_size) -> None:
        self.W = window_size
    
    def __call__(self, body_model, body_params, infos):
        poses = body_params['poses']
        padding_before = poses[:1].copy().repeat(self.W, 0)
        padding_after = poses[-1:].copy().repeat(self.W, 0)
        mean = poses.copy()
        nFrames = mean.shape[0]
        poses_full = np.vstack([padding_before, poses, padding_after])
        for w in range(1, self.W+1):
            mean += poses_full[self.W-w:self.W-w+nFrames]
            mean += poses_full[self.W+w:self.W+w+nFrames]
        mean /= 2*self.W + 1
        body_params['poses'] = mean
        return body_params