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


class NonLocalAttention(nn.Module):
    def __init__(
            self,
            in_channels=256,
            out_channels=256,
    ):
        super(NonLocalAttention, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        '''
        input [N, Feats, J, 1]
        output [N, Feats, J, 1]
        '''
        batch_size, n_feats, n_joints, _ = input.shape
        input = input.squeeze(-1)

        # Compute attention weights
        attention = torch.matmul(input.transpose(2, 1), input)
        norm_attention = F.softmax(attention, dim=-1)

        # Compute final dot product
        out = torch.matmul(input, norm_attention)
        out = self.conv1x1(out)

        out = out.unsqueeze(-1) # [N, F, J, 1]
        return out


if __name__ == '__main__':
    nla = NonLocalAttention()

    inp = torch.rand(32, 256, 24, 1)

    out = nla(inp)
    print(out.shape)