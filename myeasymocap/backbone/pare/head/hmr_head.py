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

import math
import torch
import numpy as np
import torch.nn as nn

from ..config import SMPL_MEAN_PARAMS
from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d

BN_MOMENTUM = 0.1


class HMRHead(nn.Module):
    def __init__(
            self,
            num_input_features,
            smpl_mean_params=SMPL_MEAN_PARAMS,
            estimate_var=False,
            use_separate_var_branch=False,
            uncertainty_activation='',
            backbone='resnet50',
            use_cam_feats=False,
    ):
        super(HMRHead, self).__init__()

        npose = 24 * 6
        self.npose = npose
        self.estimate_var = estimate_var
        self.use_separate_var_branch = use_separate_var_branch
        self.uncertainty_activation = uncertainty_activation
        self.backbone = backbone
        self.num_input_features = num_input_features
        self.use_cam_feats = use_cam_feats

        if use_cam_feats:
            num_input_features += 7 # 6d rotmat + vfov

        self.avgpool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(num_input_features + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        if self.estimate_var:
            # estimate variance for pose and shape parameters
            if self.use_separate_var_branch:
                # Decouple var estimation layer using separate linear layers
                self.decpose = nn.Linear(1024, npose)
                self.decshape = nn.Linear(1024, 10)
                self.deccam = nn.Linear(1024, 3)
                self.decpose_var = nn.Linear(1024, npose)
                self.decshape_var = nn.Linear(1024, 10)
                nn.init.xavier_uniform_(self.decpose_var.weight, gain=0.01)
                nn.init.xavier_uniform_(self.decshape_var.weight, gain=0.01)
            else:
                # double the output sizes to estimate var
                self.decpose = nn.Linear(1024, npose * 2)
                self.decshape = nn.Linear(1024, 10 * 2)
                self.deccam = nn.Linear(1024, 3)
        else:
            self.decpose = nn.Linear(1024, npose)
            self.decshape = nn.Linear(1024, 10)
            self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if self.backbone.startswith('hrnet'):
            self.downsample_module = self._make_head()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_head(self):
        # downsampling modules
        downsamp_modules = []
        for i in range(3):
            in_channels = self.num_input_features
            out_channels = self.num_input_features

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)

        downsamp_modules = nn.Sequential(*downsamp_modules)

        return downsamp_modules

    def forward(
            self,
            features,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            cam_rotmat=None,
            cam_vfov=None,
            n_iter=3
    ):
        # if self.backbone.startswith('hrnet'):
        #     features = self.downsample_module(features)

        batch_size = features.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.avgpool(features)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            if self.use_cam_feats:
                xc = torch.cat([xf, pred_pose, pred_shape, pred_cam,
                                rotmat_to_rot6d(cam_rotmat), cam_vfov.unsqueeze(-1)], 1)
            else:
                xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            if self.estimate_var:
                pred_pose = self.decpose(xc)[:,:self.npose] + pred_pose
                pred_shape = self.decshape(xc)[:,:10] + pred_shape
                pred_cam = self.deccam(xc) + pred_cam

                if self.use_separate_var_branch:
                    pred_pose_var = self.decpose_var(xc)
                    pred_shape_var = self.decshape_var(xc)
                else:
                    pred_pose_var = self.decpose(xc)[:,self.npose:]
                    pred_shape_var = self.decshape(xc)[:,10:]

                if self.uncertainty_activation != '':
                    # Use an activation layer to output uncertainty
                    pred_pose_var = eval(f'F.{self.uncertainty_activation}')(pred_pose_var)
                    pred_shape_var = eval(f'F.{self.uncertainty_activation}')(pred_shape_var)
            else:
                pred_pose = self.decpose(xc) + pred_pose
                pred_shape = self.decshape(xc) + pred_shape
                pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
        }

        if self.estimate_var:
            output.update({
                'pred_pose_var': torch.cat([pred_pose, pred_pose_var], dim=1),
                'pred_shape_var': torch.cat([pred_shape, pred_shape_var], dim=1),
            })

        return output

def keep_variance(x, min_variance):
    return x + min_variance