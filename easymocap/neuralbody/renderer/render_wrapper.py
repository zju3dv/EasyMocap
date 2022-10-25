'''
  @ Date: 2021-09-05 20:24:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 21:25:08
  @ FilePath: /EasyMocap/easymocap/neuralbody/renderer/render_wrapper.py
'''
import torch
import torch.nn as nn
from ...config import load_object

class RenderWrapper(nn.Module):
    def __init__(self, net, renderer_module, renderer_args, loss, loss_reg={}):
        super().__init__()
        renderer_args = dict(renderer_args)
        renderer_args['net'] = net
        self.renderer = load_object(renderer_module, renderer_args)
        self.weights = {key:val['weight'] for key, val in loss.items()}
        self.weights.update({key:val['weight'] for key, val in loss_reg.items()})
        loss = {key:load_object(val.module, val.args) for key, val in loss.items()}
        loss_reg = {key:load_object(val.module, val.args) for key, val in loss_reg.items()}
        self.loss = nn.ModuleDict(loss)
        self.loss_reg = nn.ModuleDict(loss_reg)

    def forward(self, batch):
        ret = self.renderer(batch)
        loss = 0
        scalar_stats = {}
        for key, func in self.loss.items():
            val = func(batch, ret)
            scalar_stats[key] = val
            loss += self.weights[key] * val
        for key, func in self.loss_reg.items():
            val = func(self.renderer.net, batch, ret)
            scalar_stats[key] = val
            loss += self.weights[key] * val
        for key in ['rgb_map', 'acc_map', 'occ_object', 'occ_back', 'human_0_occ']:
            if key not in ret.keys():
                continue
            scalar_stats[key] = ret[key].mean()
        for key in ['rgb']:
            if key not in ret.keys():
                continue
            scalar_stats['mean_'+key] = batch[key].mean()        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats