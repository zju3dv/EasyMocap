'''
  @ Date: 2021-09-05 20:11:27
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 20:11:27
  @ FilePath: /EasyMocap/easymocap/neuralbody/trainer/recorder.py
'''
from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os

from termcolor import colored


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

class Recorder(object):
    def __init__(self, local_rank=0, resume=False, log_dir="", task=""):
        self.local_rank = local_rank
        if local_rank > 0:
            return
        if not resume:
            print(colored('[{}] remove contents of directory {}'.format(local_rank, log_dir), 'red'))
            os.system('rm -r %s/*' % log_dir)
        # self.writer = SummaryWriter(log_dir=log_dir, flush_secs=5)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        # if 'process_' + cfg.task in globals():
        #     self.processor = globals()['process_' + cfg.task]
        # else:
        #     self.processor = None
        self.processor = None
        # self.cfg = cfg

    def update_loss_stats(self, loss_dict):
        if self.local_rank > 0:
            return
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if self.local_rank > 0:
            return
        # if self.processor is None:
        #     return
        # image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v #.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if self.local_rank > 0:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats
        image_stats = image_stats if image_stats else self.image_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)
        for k, v in image_stats.items():
            if len(v.shape) == 2:
                self.writer.add_image(pattern.format(k), v, step, dataformats='HW')
            else:
                self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        if self.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if self.local_rank > 0:
            return
        self.step = scalar_dict['step']

    def __str__(self):
        if self.local_rank > 0:
            return
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)
    
    def write_cfg(self, cfg):
        if self.local_rank > 0:
            return
        print(cfg, file=open(os.path.join(self.log_dir, 'exp.yml'), 'w'))