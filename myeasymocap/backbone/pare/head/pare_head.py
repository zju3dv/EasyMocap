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

from ..config import SMPL_MEAN_PARAMS
from ..layers.coattention import CoAttention
from ..utils.geometry import rot6d_to_rotmat, get_coord_maps
from ..utils.kp_utils import get_smpl_neighbor_triplets
from ..layers.softargmax import softargmax2d, get_heatmap_preds
from ..layers import LocallyConnected2d, KeypointAttention, interpolate
from ..layers.non_local import dot_product
from ..backbone.resnet import conv3x3, conv1x1, BasicBlock

class logger:
    @staticmethod
    def info(*args, **kwargs):
        pass
BN_MOMENTUM = 0.1


class PareHead(nn.Module):
    def __init__(
            self,
            num_joints,
            num_input_features,
            softmax_temp=1.0,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            num_camera_params=3,
            num_features_smpl=64,
            final_conv_kernel=1,
            iterative_regression=False,
            iter_residual=False,
            num_iterations=3,
            shape_input_type='feats', # 'feats.pose.shape.cam'
            pose_input_type='feats', # 'feats.neighbor_pose_feats.all_pose.self_pose.neighbor_pose.shape.cam'
            pose_mlp_num_layers=1,
            shape_mlp_num_layers=1,
            pose_mlp_hidden_size=256,
            shape_mlp_hidden_size=256,
            use_keypoint_features_for_smpl_regression=False,
            use_heatmaps='',
            use_keypoint_attention=False,
            use_postconv_keypoint_attention=False,
            keypoint_attention_act='softmax',
            use_scale_keypoint_attention=False,
            use_branch_nonlocal=None, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            use_final_nonlocal=None, # 'concatenation', 'dot_product', 'embedded_gaussian', 'gaussian'
            backbone='resnet',
            use_hmr_regression=False,
            use_coattention=False,
            num_coattention_iter=1,
            coattention_conv='simple', # 'double_1', 'double_3', 'single_1', 'single_3', 'simple'
            use_upsampling=False,
            use_soft_attention=False, # Stefan & Otmar 3DV style attention
            num_branch_iteration=0,
            branch_deeper=False,
            use_resnet_conv_hrnet=False,
            use_position_encodings=None,
            use_mean_camshape=False,
            use_mean_pose=False,
            init_xavier=False,
    ):
        super(PareHead, self).__init__()
        self.backbone = backbone
        self.num_joints = num_joints
        self.deconv_with_bias = False
        self.use_heatmaps = use_heatmaps
        self.num_iterations = num_iterations
        self.use_final_nonlocal = use_final_nonlocal
        self.use_branch_nonlocal = use_branch_nonlocal
        self.use_hmr_regression = use_hmr_regression
        self.use_coattention = use_coattention
        self.num_coattention_iter = num_coattention_iter
        self.coattention_conv = coattention_conv
        self.use_soft_attention = use_soft_attention
        self.num_branch_iteration = num_branch_iteration
        self.iter_residual = iter_residual
        self.iterative_regression = iterative_regression
        self.pose_mlp_num_layers = pose_mlp_num_layers
        self.shape_mlp_num_layers = shape_mlp_num_layers
        self.pose_mlp_hidden_size = pose_mlp_hidden_size
        self.shape_mlp_hidden_size = shape_mlp_hidden_size
        self.use_keypoint_attention = use_keypoint_attention
        self.use_keypoint_features_for_smpl_regression = use_keypoint_features_for_smpl_regression
        self.use_position_encodings = use_position_encodings
        self.use_mean_camshape = use_mean_camshape
        self.use_mean_pose = use_mean_pose

        self.num_input_features = num_input_features

        if use_soft_attention:
            # These options should be True by default when soft attention is used
            self.use_keypoint_features_for_smpl_regression = True
            self.use_hmr_regression = True
            self.use_coattention = False
            logger.warning('Coattention cannot be used together with soft attention')
            logger.warning('Overriding use_coattention=False')

        if use_coattention:
            self.use_keypoint_features_for_smpl_regression = False
            logger.warning('\"use_keypoint_features_for_smpl_regression\" cannot be used together with co-attention')
            logger.warning('Overriding \"use_keypoint_features_for_smpl_regression\"=False')

        if use_hmr_regression:
            self.iterative_regression = False
            logger.warning('iterative_regression cannot be used together with hmr regression')

        if self.use_heatmaps in ['part_segm', 'attention']:
            logger.info('\"Keypoint Attention\" should be activated to be able to use part segmentation')
            logger.info('Overriding use_keypoint_attention')
            self.use_keypoint_attention = True

        assert num_iterations > 0, '\"num_iterations\" should be greater than 0.'

        if use_position_encodings:
            assert backbone.startswith('hrnet'), 'backbone should be hrnet to use position encodings'
            # self.pos_enc = get_coord_maps(size=56)
            self.register_buffer('pos_enc', get_coord_maps(size=56))
            num_input_features += 2
            self.num_input_features = num_input_features

        if backbone.startswith('hrnet'):
            if use_resnet_conv_hrnet:
                logger.info('Using resnet block for keypoint and smpl conv layers...')
                self.keypoint_deconv_layers = self._make_res_conv_layers(
                    input_channels=self.num_input_features,
                    num_channels=num_deconv_filters[-1],
                    num_basic_blocks=num_deconv_layers,
                )
                self.num_input_features = num_input_features
                self.smpl_deconv_layers = self._make_res_conv_layers(
                    input_channels=self.num_input_features,
                    num_channels=num_deconv_filters[-1],
                    num_basic_blocks=num_deconv_layers,
                )
            else:
                self.keypoint_deconv_layers = self._make_conv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    (3,)*num_deconv_layers,
                )
                self.num_input_features = num_input_features
                self.smpl_deconv_layers = self._make_conv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    (3,)*num_deconv_layers,
                )
        else:
            # part branch that estimates 2d keypoints

            conv_fn = self._make_upsample_layer if use_upsampling else self._make_deconv_layer

            if use_upsampling:
                logger.info('Upsampling is active to increase spatial dimension')
                logger.info(f'Upsampling conv kernels: {num_deconv_kernels}')

            self.keypoint_deconv_layers = conv_fn(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
            # reset inplanes to 2048 -> final resnet layer
            self.num_input_features = num_input_features
            self.smpl_deconv_layers = conv_fn(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )

        pose_mlp_inp_dim = num_deconv_filters[-1]
        smpl_final_dim = num_features_smpl
        shape_mlp_inp_dim = num_joints * smpl_final_dim

        if self.use_soft_attention:
            logger.info('Soft attention (Stefan & Otmar 3DV) is active')
            self.keypoint_final_layer = nn.Sequential(
                conv3x3(num_deconv_filters[-1], 256),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                conv1x1(256, num_joints+1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints),
            )

            soft_att_feature_size = smpl_final_dim # if use_hmr_regression else pose_mlp_inp_dim
            self.smpl_final_layer = nn.Sequential(
                conv3x3(num_deconv_filters[-1], 256),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                conv1x1(256, soft_att_feature_size),
            )
            # pose_mlp_inp_dim = soft_att_feature_size
        else:
            self.keypoint_final_layer = nn.Conv2d(
                in_channels=num_deconv_filters[-1],
                out_channels=num_joints+1 if self.use_heatmaps in ('part_segm', 'part_segm_pool') else num_joints,
                kernel_size=final_conv_kernel,
                stride=1,
                padding=1 if final_conv_kernel == 3 else 0,
            )

            self.smpl_final_layer = nn.Conv2d(
                in_channels=num_deconv_filters[-1],
                out_channels=smpl_final_dim,
                kernel_size=final_conv_kernel,
                stride=1,
                padding=1 if final_conv_kernel == 3 else 0,
            )

        # temperature for softargmax function
        self.register_buffer('temperature', torch.tensor(softmax_temp))

        # if self.iterative_regression or self.num_branch_iteration > 0 or self.use_coattention:
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        if self.iterative_regression:
            # enable iterative regression similar to HMR
            # these are the features that can be used as input to final MLPs
            input_type_dim = {
                'feats': 0, # image features for self
                'neighbor_pose_feats': 2 * 256, # image features from neighbor joints
                'all_pose': 24 * 6, # rot6d of all joints from previous iter
                'self_pose': 6, # rot6d of self
                'neighbor_pose': 2 * 6, # rot6d of neighbor joints from previous iter
                'shape': 10, # smpl betas/shape
                'cam': num_camera_params, # weak perspective camera
            }

            assert 'feats' in shape_input_type, '\"feats\" should be the default value'
            assert 'feats' in pose_input_type, '\"feats\" should be the default value'

            self.shape_input_type = shape_input_type.split('.')
            self.pose_input_type = pose_input_type.split('.')

            pose_mlp_inp_dim = pose_mlp_inp_dim + sum([input_type_dim[x] for x in self.pose_input_type])
            shape_mlp_inp_dim = shape_mlp_inp_dim + sum([input_type_dim[x] for x in self.shape_input_type])

            logger.debug(f'Shape MLP takes \"{self.shape_input_type}\" as input, '
                         f'input dim: {shape_mlp_inp_dim}')
            logger.debug(f'Pose MLP takes \"{self.pose_input_type}\" as input, '
                         f'input dim: {pose_mlp_inp_dim}')

        self.pose_mlp_inp_dim = pose_mlp_inp_dim
        self.shape_mlp_inp_dim = shape_mlp_inp_dim

        if self.use_hmr_regression:
            logger.info(f'HMR regression is active...')
            # enable iterative regression similar to HMR

            self.fc1 = nn.Linear(num_joints * smpl_final_dim + (num_joints * 6) + 10 + num_camera_params, 1024)
            self.drop1 = nn.Dropout()
            self.fc2 = nn.Linear(1024, 1024)
            self.drop2 = nn.Dropout()
            self.decpose = nn.Linear(1024, (num_joints * 6))
            self.decshape = nn.Linear(1024, 10)
            self.deccam = nn.Linear(1024, num_camera_params)

            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        else:
            # here we use 2 different MLPs to estimate shape and camera
            # They take a channelwise downsampled version of smpl features
            self.shape_mlp = self._get_shape_mlp(output_size=10)
            self.cam_mlp = self._get_shape_mlp(output_size=num_camera_params)

            # for pose each joint has a separate MLP
            # weights for these MLPs are not shared
            # hence we use Locally Connected layers
            # TODO support kernel_size > 1 to access context of other joints
            self.pose_mlp = self._get_pose_mlp(num_joints=num_joints, output_size=6)

            if init_xavier:
                nn.init.xavier_uniform_(self.shape_mlp.weight, gain=0.01)
                nn.init.xavier_uniform_(self.cam_mlp.weight, gain=0.01)
                nn.init.xavier_uniform_(self.pose_mlp.weight, gain=0.01)

        if self.use_branch_nonlocal:
            logger.info(f'Branch nonlocal is active, type {self.use_branch_nonlocal}')
            self.branch_2d_nonlocal = eval(self.use_branch_nonlocal).NONLocalBlock2D(
                in_channels=num_deconv_filters[-1],
                sub_sample=False,
                bn_layer=True,
            )

            self.branch_3d_nonlocal = eval(self.use_branch_nonlocal).NONLocalBlock2D(
                in_channels=num_deconv_filters[-1],
                sub_sample=False,
                bn_layer=True,
            )

        if self.use_final_nonlocal:
            logger.info(f'Final nonlocal is active, type {self.use_final_nonlocal}')
            self.final_pose_nonlocal = eval(self.use_final_nonlocal).NONLocalBlock1D(
                in_channels=self.pose_mlp_inp_dim,
                sub_sample=False,
                bn_layer=True,
            )

            self.final_shape_nonlocal = eval(self.use_final_nonlocal).NONLocalBlock1D(
                in_channels=num_features_smpl,
                sub_sample=False,
                bn_layer=True,
            )

        if self.use_keypoint_attention:
            logger.info('Keypoint attention is active')
            self.keypoint_attention = KeypointAttention(
                use_conv=use_postconv_keypoint_attention,
                in_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
                out_channels=(self.pose_mlp_inp_dim, smpl_final_dim),
                act=keypoint_attention_act,
                use_scale=use_scale_keypoint_attention,
            )

        if self.use_coattention:
            logger.info(f'Coattention is active, final conv type {self.coattention_conv}')
            self.coattention = CoAttention(n_channel=num_deconv_filters[-1], final_conv=self.coattention_conv)

        if self.num_branch_iteration > 0:
            logger.info(f'Branch iteration is active')
            if branch_deeper:
                self.branch_iter_2d_nonlocal = nn.Sequential(
                    conv3x3(num_deconv_filters[-1], 256),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    dot_product.NONLocalBlock2D(
                        in_channels=num_deconv_filters[-1],
                        sub_sample=False,
                        bn_layer=True,
                    )
                )

                self.branch_iter_3d_nonlocal = nn.Sequential(
                    conv3x3(num_deconv_filters[-1], 256),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    dot_product.NONLocalBlock2D(
                        in_channels=num_deconv_filters[-1],
                        sub_sample=False,
                        bn_layer=True,
                    )
                )
            else:
                self.branch_iter_2d_nonlocal = dot_product.NONLocalBlock2D(
                    in_channels=num_deconv_filters[-1],
                    sub_sample=False,
                    bn_layer=True,
                )

                self.branch_iter_3d_nonlocal = dot_product.NONLocalBlock2D(
                    in_channels=num_deconv_filters[-1],
                    sub_sample=False,
                    bn_layer=True,
                )

    def _get_shape_mlp(self, output_size):
        if self.shape_mlp_num_layers == 1:
            return nn.Linear(self.shape_mlp_inp_dim, output_size)

        module_list = []
        for i in range(self.shape_mlp_num_layers):
            if i == 0:
                module_list.append(
                    nn.Linear(self.shape_mlp_inp_dim, self.shape_mlp_hidden_size)
                )
            elif i == self.shape_mlp_num_layers - 1:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, output_size)
                )
            else:
                module_list.append(
                    nn.Linear(self.shape_mlp_hidden_size, self.shape_mlp_hidden_size)
                )
        return nn.Sequential(*module_list)

    def _get_pose_mlp(self, num_joints, output_size):
        if self.pose_mlp_num_layers == 1:
            return LocallyConnected2d(
                in_channels=self.pose_mlp_inp_dim,
                out_channels=output_size,
                output_size=[num_joints, 1],
                kernel_size=1,
                stride=1,
            )

        module_list = []
        for i in range(self.pose_mlp_num_layers):
            if i == 0:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_inp_dim,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            elif i == self.pose_mlp_num_layers - 1:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=output_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
            else:
                module_list.append(
                    LocallyConnected2d(
                        in_channels=self.pose_mlp_hidden_size,
                        out_channels=self.pose_mlp_hidden_size,
                        output_size=[num_joints, 1],
                        kernel_size=1,
                        stride=1,
                    )
                )
        return nn.Sequential(*module_list)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_res_conv_layers(self, input_channels, num_channels=64,
                              num_heads=1, num_basic_blocks=2):
        head_layers = []

        # kernel_sizes, strides, paddings = self._get_trans_cfg()
        # for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
        head_layers.append(nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        )

        for i in range(num_heads):
            layers = []
            for _ in range(num_basic_blocks):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        # head_layers.append(nn.Conv2d(in_channels=num_channels, out_channels=output_channels,
        #                              kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*head_layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _make_upsample_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_layers is different len(num_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_layers is different len(num_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(
                nn.Conv2d(in_channels=self.num_input_features, out_channels=planes,
                          kernel_size=kernel, stride=1, padding=padding, bias=self.deconv_with_bias)
            )
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            # if self.use_self_attention:
            #     layers.append(SelfAttention(planes))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _prepare_pose_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[0], pred_pose.shape[2]

        joint_triplets = get_smpl_neighbor_triplets()

        inp_list = []

        for inp_type in self.pose_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'neighbor_pose_feats':
                # add the image features from neighboring joints
                n_pose_feat = []
                for jt in joint_triplets:
                    n_pose_feat.append(
                        feats[:, :, jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose_feat = torch.cat(n_pose_feat, 2)
                inp_list.append(n_pose_feat)

            if inp_type == 'self_pose':
                # add image features
                inp_list.append(pred_pose)

            if inp_type == 'all_pose':
                # append all of the joint angels
                all_pose = pred_pose.reshape(batch_size, -1, 1)[..., None].repeat(1, 1, num_joints, 1)
                inp_list.append(all_pose)

            if inp_type == 'neighbor_pose':
                # append only the joint angles of neighboring ones
                n_pose = []
                for jt in joint_triplets:
                    n_pose.append(
                        pred_pose[:,:,jt[1:]].reshape(batch_size, -1, 1).unsqueeze(-2)
                    )
                n_pose = torch.cat(n_pose, 2)
                inp_list.append(n_pose)

            if inp_type == 'shape':
                # append shape predictions
                pred_shape = pred_shape[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                pred_cam = pred_cam[..., None, None].repeat(1, 1, num_joints, 1)
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        # for i,inp in enumerate(inp_list):
        #     print(i, inp.shape)

        return torch.cat(inp_list, 1)

    def _prepare_shape_mlp_inp(self, feats, pred_pose, pred_shape, pred_cam):
        # feats shape: [N, 256, J, 1]
        # pose shape: [N, 6, J, 1]
        # cam shape: [N, 3]
        # beta shape: [N, 10]
        batch_size, num_joints = pred_pose.shape[:2]

        inp_list = []

        for inp_type in self.shape_input_type:
            if inp_type == 'feats':
                # add image features
                inp_list.append(feats)

            if inp_type == 'all_pose':
                # append all of the joint angels
                pred_pose = pred_pose.reshape(batch_size, -1)
                inp_list.append(pred_pose)

            if inp_type == 'shape':
                # append shape predictions
                inp_list.append(pred_shape)

            if inp_type == 'cam':
                # append camera predictions
                inp_list.append(pred_cam)

        assert len(inp_list) > 0

        return torch.cat(inp_list, 1)

    def forward(self, features, gt_segm=None):
        batch_size = features.shape[0]

        init_pose = self.init_pose.expand(batch_size, -1)  # N, Jx6
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if self.use_position_encodings:
            features = torch.cat((features, self.pos_enc.repeat(features.shape[0], 1, 1, 1)), 1)

        output = {}

        ############## 2D PART BRANCH FEATURES ##############
        part_feats = self._get_2d_branch_feats(features)

        ############## GET PART ATTENTION MAP ##############
        part_attention = self._get_part_attention_map(part_feats, output)

        ############## 3D SMPL BRANCH FEATURES ##############
        smpl_feats = self._get_3d_smpl_feats(features, part_feats)

        ############## SAMPLE LOCAL FEATURES ##############
        if gt_segm is not None:
            # logger.debug(gt_segm.shape)
            # import IPython; IPython.embed(); exit()
            gt_segm = F.interpolate(gt_segm.unsqueeze(1).float(), scale_factor=(1/4, 1/4), mode='nearest').long().squeeze(1)
            part_attention = F.one_hot(gt_segm.to('cpu'), num_classes=self.num_joints + 1).permute(0,3,1,2).float()[:,1:,:,:]
            part_attention = part_attention.to('cuda')
            # part_attention = F.interpolate(part_attention, scale_factor=1/4, mode='bilinear', align_corners=True)
            # import IPython; IPython.embed(); exit()
        point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)

        ############## GET FINAL PREDICTIONS ##############
        pred_pose, pred_shape, pred_cam = self._get_final_preds(
            point_local_feat, cam_shape_feats, init_pose, init_shape, init_cam
        )

        if self.use_coattention:
            for c in range(self.num_coattention_iter):
                smpl_feats, part_feats = self.coattention(smpl_feats, part_feats)
                part_attention = self._get_part_attention_map(part_feats, output)
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
                pred_pose, pred_shape, pred_cam = self._get_final_preds(
                    point_local_feat, cam_shape_feats, pred_pose, pred_shape, pred_cam
                )

        if self.num_branch_iteration > 0:
            for nbi in range(self.num_branch_iteration):
                if self.use_soft_attention:
                    smpl_feats = self.branch_iter_3d_nonlocal(smpl_feats)
                    part_feats = self.branch_iter_2d_nonlocal(part_feats)
                else:
                    smpl_feats = self.branch_iter_3d_nonlocal(smpl_feats)
                    part_feats = smpl_feats

                part_attention = self._get_part_attention_map(part_feats, output)
                point_local_feat, cam_shape_feats = self._get_local_feats(smpl_feats, part_attention, output)
                pred_pose, pred_shape, pred_cam = self._get_final_preds(
                    point_local_feat, cam_shape_feats, pred_pose, pred_shape, pred_cam,
                )

        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)

        output.update({
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        })
        return output

    def _get_local_feats(self, smpl_feats, part_attention, output):
        cam_shape_feats = self.smpl_final_layer(smpl_feats)

        if self.use_keypoint_attention:
            point_local_feat = self.keypoint_attention(smpl_feats, part_attention)
            cam_shape_feats = self.keypoint_attention(cam_shape_feats, part_attention)
        else:
            point_local_feat = interpolate(smpl_feats, output['pred_kp2d'])
            cam_shape_feats = interpolate(cam_shape_feats, output['pred_kp2d'])
        return point_local_feat, cam_shape_feats

    def _get_2d_branch_feats(self, features):
        part_feats = self.keypoint_deconv_layers(features)
        if self.use_branch_nonlocal:
            part_feats = self.branch_2d_nonlocal(part_feats)
        return part_feats

    def _get_3d_smpl_feats(self, features, part_feats):
        if self.use_keypoint_features_for_smpl_regression:
            smpl_feats = part_feats
        else:
            smpl_feats = self.smpl_deconv_layers(features)
            if self.use_branch_nonlocal:
                smpl_feats = self.branch_3d_nonlocal(smpl_feats)

        return smpl_feats

    def _get_part_attention_map(self, part_feats, output):
        heatmaps = self.keypoint_final_layer(part_feats)

        if self.use_heatmaps == 'hm':
            # returns coords between [-1,1]
            pred_kp2d, confidence = get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
            output['pred_kp2d_conf'] = confidence
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'hm_soft':
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        elif self.use_heatmaps == 'part_segm':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:,1:,:,:] # remove the first channel which encodes the background
        elif self.use_heatmaps == 'part_segm_pool':
            output['pred_segm_mask'] = heatmaps
            heatmaps = heatmaps[:,1:,:,:] # remove the first channel which encodes the background
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature) # get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d

            for k, v in output.items():
                if torch.any(torch.isnan(v)):
                    logger.debug(f'{k} is Nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if torch.any(torch.isinf(v)):
                    logger.debug(f'{k} is Inf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # if torch.any(torch.isnan(pred_kp2d)):
            #     print('pred_kp2d nan', pred_kp2d.min(), pred_kp2d.max())
            # if torch.any(torch.isnan(heatmaps)):
            #     print('heatmap nan', heatmaps.min(), heatmaps.max())
            #
            # if torch.any(torch.isinf(pred_kp2d)):
            #     print('pred_kp2d inf', pred_kp2d.min(), pred_kp2d.max())
            # if torch.any(torch.isinf(heatmaps)):
            #     print('heatmap inf', heatmaps.min(), heatmaps.max())

        elif self.use_heatmaps == 'attention':
            output['pred_attention'] = heatmaps
        else:
            # returns coords between [-1,1]
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
            output['pred_heatmaps_2d'] = heatmaps
        return heatmaps

    def _get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam):
        if self.use_hmr_regression:
            return self._hmr_get_final_preds(cam_shape_feats, init_pose, init_shape, init_cam)
        else:
            return self._pare_get_final_preds(pose_feats, cam_shape_feats, init_pose, init_shape, init_cam)

    def _hmr_get_final_preds(self, cam_shape_feats, init_pose, init_shape, init_cam):
        if self.use_final_nonlocal:
            cam_shape_feats = self.final_shape_nonlocal(cam_shape_feats)

        xf = torch.flatten(cam_shape_feats, start_dim=1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(3):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        return pred_pose, pred_shape, pred_cam

    def _pare_get_final_preds(self, pose_feats, cam_shape_feats, init_pose, init_shape, init_cam):
        pose_feats = pose_feats.unsqueeze(-1)  #

        if init_pose.shape[-1] == 6:
            # This means init_pose comes from a previous iteration
            init_pose = init_pose.transpose(2,1).unsqueeze(-1)
        else:
            # This means init pose comes from mean pose
            init_pose = init_pose.reshape(init_pose.shape[0], 6, -1).unsqueeze(-1)

        if self.iterative_regression:

            shape_feats = torch.flatten(cam_shape_feats, start_dim=1)

            pred_pose = init_pose  # [N, 6, J, 1]
            pred_cam = init_cam  # [N, 3]
            pred_shape = init_shape  # [N, 10]

            # import IPython; IPython.embed(); exit(1)

            for i in range(self.num_iterations):
                # pose_feats shape: [N, 256, 24, 1]
                # shape_feats shape: [N, 24*64]
                pose_mlp_inp = self._prepare_pose_mlp_inp(pose_feats, pred_pose, pred_shape, pred_cam)
                shape_mlp_inp = self._prepare_shape_mlp_inp(shape_feats, pred_pose, pred_shape, pred_cam)

                # print('pose_mlp_inp', pose_mlp_inp.shape)
                # print('shape_mlp_inp', shape_mlp_inp.shape)
                # TODO: this does not work but let it go since we dont use iterative regression for now.
                # if self.use_final_nonlocal:
                #     pose_mlp_inp = self.final_pose_nonlocal(pose_mlp_inp)
                #     shape_mlp_inp = self.final_shape_nonlocal(shape_mlp_inp)

                if self.iter_residual:
                    pred_pose = self.pose_mlp(pose_mlp_inp) + pred_pose
                    pred_cam = self.cam_mlp(shape_mlp_inp) + pred_cam
                    pred_shape = self.shape_mlp(shape_mlp_inp) + pred_shape
                else:
                    pred_pose = self.pose_mlp(pose_mlp_inp)
                    pred_cam = self.cam_mlp(shape_mlp_inp)
                    pred_shape = self.shape_mlp(shape_mlp_inp) + init_shape
        else:
            shape_feats = cam_shape_feats
            if self.use_final_nonlocal:
                pose_feats = self.final_pose_nonlocal(pose_feats.squeeze(-1)).unsqueeze(-1)
                shape_feats = self.final_shape_nonlocal(shape_feats)

            shape_feats = torch.flatten(shape_feats, start_dim=1)

            pred_pose = self.pose_mlp(pose_feats)
            pred_cam = self.cam_mlp(shape_feats)
            pred_shape = self.shape_mlp(shape_feats)

            if self.use_mean_camshape:
                pred_cam = pred_cam + init_cam
                pred_shape = pred_shape + init_shape

            if self.use_mean_pose:
                pred_pose = pred_pose + init_pose


        pred_pose = pred_pose.squeeze(-1).transpose(2, 1) # N, J, 6
        return pred_pose, pred_shape, pred_cam

    def forward_pretraining(self, features):
        # TODO: implement pretraining
        kp_feats = self.keypoint_deconv_layers(features)
        heatmaps = self.keypoint_final_layer(kp_feats)

        output = {}

        if self.use_heatmaps == 'hm':
            # returns coords between [-1,1]
            pred_kp2d, confidence = get_heatmap_preds(heatmaps)
            output['pred_kp2d'] = pred_kp2d
            output['pred_kp2d_conf'] = confidence
        elif self.use_heatmaps == 'hm_soft':
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d
        else:
            # returns coords between [-1,1]
            pred_kp2d, _ = softargmax2d(heatmaps, self.temperature)
            output['pred_kp2d'] = pred_kp2d

        if self.use_keypoint_features_for_smpl_regression:
            smpl_feats = kp_feats
        else:
            smpl_feats = self.smpl_deconv_layers(features)

        cam_shape_feats = self.smpl_final_layer(smpl_feats)

        output.update({
            'kp_feats': heatmaps,
            'heatmaps': heatmaps,
            'smpl_feats': smpl_feats,
            'cam_shape_feats': cam_shape_feats,
        })
        return output