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

def get_backbone_info(backbone):
    info = {
        'resnet18': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet34': {'n_output_channels': 512, 'downsample_rate': 4},
        'resnet50': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_adf_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet50_dropout': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet101': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnet152': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext50_32x4d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'resnext101_32x8d': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet50_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'wide_resnet101_2': {'n_output_channels': 2048, 'downsample_rate': 4},
        'mobilenet_v2': {'n_output_channels': 1280, 'downsample_rate': 4},
        'hrnet_w32': {'n_output_channels': 480, 'downsample_rate': 4},
        'hrnet_w48': {'n_output_channels': 720, 'downsample_rate': 4},
        # 'hrnet_w64': {'n_output_channels': 2048, 'downsample_rate': 4},
        'dla34': {'n_output_channels': 512, 'downsample_rate': 4},
    }
    return info[backbone]