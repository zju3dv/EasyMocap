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
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.resnet import conv1x1, conv3x3


class CoAttention(nn.Module):
    def __init__(
            self,
            n_channel,
            final_conv='simple', # 'double_1', 'double_3', 'single_1', 'single_3', 'simple'
    ):
        super(CoAttention, self).__init__()
        self.linear_e = nn.Linear(n_channel, n_channel, bias=False)
        self.channel = n_channel
        # self.dim = all_dim
        self.gate = nn.Conv2d(n_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.softmax = nn.Sigmoid()

        if final_conv.startswith('double'):
            kernel_size = int(final_conv[-1])
            conv = conv1x1 if kernel_size == 1 else conv3x3
            self.final_conv_1 = nn.Sequential(
                conv(n_channel * 2, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
                conv(n_channel, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            )
            self.final_conv_2 = nn.Sequential(
                conv(n_channel * 2, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
                conv(n_channel, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            )
        elif final_conv.startswith('single'):
            kernel_size = int(final_conv[-1])
            conv = conv1x1 if kernel_size == 1 else conv3x3
            self.final_conv_1 = nn.Sequential(
                conv(n_channel*2, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            )
            self.final_conv_2 = nn.Sequential(
                conv(n_channel*2, n_channel),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            )
        elif final_conv == 'simple':
            self.final_conv_1 = conv1x1(n_channel * 2, n_channel)
            self.final_conv_2 = conv1x1(n_channel * 2, n_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_1, input_2):
        '''
        input_1: [N, C, H, W]
        input_2: [N, C, H, W]
        '''

        b, c, h, w = input_1.shape
        exemplar, query = input_1, input_2

        exemplar_flat = exemplar.reshape(-1, c, h*w)  # N,C,H*W
        query_flat = query.reshape(-1, c, h*w)

        # Compute coattention scores, S in the paper
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1)
        exemplar_att = torch.bmm(query_flat, B)

        input1_att = exemplar_att.reshape(-1, c, h, w)
        input2_att = query_att.reshape(-1, c, h, w)

        # Apply gating on S, section gated coattention
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)

        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        # Concatenate inputs with their attended version
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)

        input1 = self.final_conv_1(input1_att)
        input2 = self.final_conv_2(input2_att)

        return input1, input2
