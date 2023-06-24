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
import torch.nn.functional as F


def _softmax(tensor, temperature, dim=-1):
    return F.softmax(tensor * temperature, dim=dim)


def softargmax1d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, dim = heatmaps.shape
    points = torch.arange(0, dim, device=device, dtype=dtype).reshape(1, 1, dim).expand(batch_size, -1, -1)
    # y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW

    # points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (normalized_heatmap.reshape(batch_size, -1, dim) * points).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints = (keypoints / (dim - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, dim)


def softargmax2d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, height, width = heatmaps.shape
    x = torch.arange(0, width, device=device, dtype=dtype).reshape(1, 1, 1, width).expand(batch_size, -1, height, -1)
    y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1).expand(batch_size, -1, -1, width)
    # Should be Bx2xHxW
    points = torch.cat([x, y], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx2
    keypoints = (
            normalized_heatmap.reshape(batch_size, -1, 1, height * width) *
            points.reshape(batch_size, 1, 2, -1)).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints[:, :, 0] = (keypoints[:, :, 0] / (width - 1) * 2 - 1)
        keypoints[:, :, 1] = (keypoints[:, :, 1] / (height - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, height, width)


def softargmax3d(
        heatmaps,
        temperature=None,
        normalize_keypoints=True,
):
    dtype, device = heatmaps.dtype, heatmaps.device
    if temperature is None:
        temperature = torch.tensor(1.0, dtype=dtype, device=device)
    batch_size, num_channels, height, width, depth = heatmaps.shape
    x = torch.arange(0, width, device=device, dtype=dtype).reshape(1, 1, 1, width, 1).expand(batch_size, -1, height, -1, depth)
    y = torch.arange(0, height, device=device, dtype=dtype).reshape(1, 1, height, 1, 1).expand(batch_size, -1, -1, width, depth)
    z = torch.arange(0, depth, device=device, dtype=dtype).reshape(1, 1, 1, 1, depth).expand(batch_size, -1, height, width, -1)
    # Should be Bx2xHxW
    points = torch.cat([x, y, z], dim=1)
    normalized_heatmap = _softmax(
        heatmaps.reshape(batch_size, num_channels, -1),
        temperature=temperature.reshape(1, -1, 1),
        dim=-1)

    # Should be BxJx3
    keypoints = (
            normalized_heatmap.reshape(batch_size, -1, 1, height * width * depth) *
            points.reshape(batch_size, 1, 3, -1)).sum(dim=-1)

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        keypoints[:, :, 0] = (keypoints[:, :, 0] / (width - 1) * 2 - 1)
        keypoints[:, :, 1] = (keypoints[:, :, 1] / (height - 1) * 2 - 1)
        keypoints[:, :, 2] = (keypoints[:, :, 2] / (depth - 1) * 2 - 1)

    return keypoints, normalized_heatmap.reshape(
        batch_size, -1, height, width, depth)


def get_heatmap_preds(batch_heatmaps, normalize_keypoints=True):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask

    if normalize_keypoints:
        # Normalize keypoints to [-1, 1]
        preds[:, :, 0] = (preds[:, :, 0] / (width - 1) * 2 - 1)
        preds[:, :, 1] = (preds[:, :, 1] / (height - 1) * 2 - 1)

    return preds, maxvals