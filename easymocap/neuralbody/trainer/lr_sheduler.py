'''
  @ Date: 2021-09-05 20:07:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 20:10:02
  @ FilePath: /EasyMocap/easymocap/neuralbody/trainer/lr_sheduler.py
'''
import torch
from collections import Counter
from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]

def Scheduler(cfg_scheduler, optimizer):
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    else:
        raise NotImplementedError
    return scheduler

def set_lr_scheduler(cfg_scheduler, scheduler):
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    scheduler.gamma = cfg_scheduler.gamma