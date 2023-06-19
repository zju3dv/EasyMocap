import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
# https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth
# https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth
from ..basetopdown import BaseTopDownModelCache, get_preds_from_heatmaps, gdown_models

class TopDownAsMMPose(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.bacbone = backbone
        self.head = head
    
    def forward(self, x):
        feat_list = self.bacbone(x)
        size = feat_list[0].shape[-2:]
        resized_inputs = [
            nn.functional.interpolate(feat, size, mode='bilinear', align_corners=False) \
            for feat in feat_list
        ]
        resized_inputs = torch.cat(resized_inputs, 1)
        out = self.head(resized_inputs)
        pred = get_preds_from_heatmaps(out.detach().cpu().numpy())
        return {'keypoints': pred}

class MyHand2D(BaseTopDownModelCache):
    def __init__(self, ckpt, url=None, mode='hrnet'):
        if mode == 'hrnet':
            super().__init__(name='hand2d', bbox_scale=1.1, res_input=256)
            from .hrnet import PoseHighResolutionNet
            backbone = PoseHighResolutionNet(inp_ch=3, out_ch=21, W=18, multi_scale_final=True, add_final_layer=False)
            checkpoint = torch.load(ckpt, map_location='cpu')['state_dict']
            self.load_checkpoint(backbone, checkpoint, prefix='backbone.', strict=True)
            head = nn.Sequential(
                nn.Conv2d(270, 270, kernel_size=1),
                nn.BatchNorm2d(270),
                nn.ReLU(inplace=True),
                nn.Conv2d(270, 21, kernel_size=1)
            )
            self.load_checkpoint(head, checkpoint, prefix='keypoint_head.final_layer.', strict=True)
            # self.model = nn.Sequential(backbone, head)
            self.model = TopDownAsMMPose(backbone, head)
        elif mode == 'resnet':
            super().__init__(name='hand2d', bbox_scale=1.1, res_input=256, mean=[0., 0., 0.], std=[1., 1., 1.])
            from .resnet import ResNet_Deconv
            if not os.path.exists(ckpt) and url is not None:
                gdown_models(ckpt, url)
            assert os.path.exists(ckpt), f'{ckpt} not exists'
            checkpoint = torch.load(ckpt, map_location='cpu')['state_dict']
            model = ResNet_Deconv()
            self.load_checkpoint(model, checkpoint, prefix='model.', strict=True)
            self.model = model
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def __call__(self, bbox, images, imgnames):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            bbox = [bbox]
            squeeze = True
        nViews = len(images)
        kpts_all = []
        for nv in range(nViews):
            if bbox[nv].shape[0] == 0:
                kpts_all.append(np.zeros((21, 3)))
                continue
            _bbox = bbox[nv]
            if len(_bbox.shape) == 1:
                _bbox = _bbox[None]
            output = super().__call__(_bbox, images[nv], imgnames[nv])
            kpts = output['params']['keypoints']
            conf = kpts[..., -1:]
            kpts = self.batch_affine_transform(kpts, output['params']['inv_trans'])
            kpts = np.concatenate([kpts, conf], axis=-1)
            if len(kpts.shape) == 3:
                kpts = kpts[0]
            kpts_all.append(kpts)
        kpts_all = np.stack(kpts_all)
        if squeeze:
            kpts_all = kpts_all[0]
        return {
            'keypoints': kpts_all
        }