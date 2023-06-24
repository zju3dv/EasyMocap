# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KeypointAttention(nn.Module):
    def __init__(self, use_conv=False, in_channels=(256, 64), out_channels=(256, 64), act='softmax', use_scale=False):
        super(KeypointAttention, self).__init__()
        self.use_conv = use_conv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.use_scale = use_scale
        if use_conv:
            self.conv1x1_pose = nn.Conv1d(in_channels[0], out_channels[0], kernel_size=1)
            self.conv1x1_shape_cam = nn.Conv1d(in_channels[1], out_channels[1], kernel_size=1)

    def forward(self, features, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape

        if self.use_scale:
            scale = 1.0 / np.sqrt(height * width)
            heatmaps = heatmaps * scale

        if self.act == 'softmax':
            normalized_heatmap = F.softmax(heatmaps.reshape(batch_size, num_joints, -1), dim=-1)
        elif self.act == 'sigmoid':
            normalized_heatmap = torch.sigmoid(heatmaps.reshape(batch_size, num_joints, -1))
        features = features.reshape(batch_size, -1, height*width)

        attended_features = torch.matmul(normalized_heatmap, features.transpose(2,1))
        attended_features = attended_features.transpose(2,1)

        if self.use_conv:
            if attended_features.shape[1] == self.in_channels[0]:
                attended_features = self.conv1x1_pose(attended_features)
            else:
                attended_features = self.conv1x1_shape_cam(attended_features)

        return attended_features