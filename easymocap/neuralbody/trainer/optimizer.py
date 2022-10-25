'''
  @ Date: 2021-09-05 20:05:42
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 20:05:42
  @ FilePath: /EasyMocap/easymocap/neuralbody/trainer/optimizer.py
'''

import torch

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def Optimizer(net, cfg):
    params = []
    lr = cfg.lr
    weight_decay = cfg.weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.optim:
        optimizer = _optimizer_factory[cfg.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.optim](params, lr, momentum=0.9)

    return optimizer