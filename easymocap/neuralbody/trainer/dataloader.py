'''
  @ Date: 2021-07-20 12:32:29
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 20:19:11
  @ FilePath: /EasyMocap/easymocap/neuralbody/trainer/dataloader.py
'''
from easymocap.config.baseconfig import load_object
import torch

def make_data_sampler(cfg, dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        from .samplers import FrameSampler
        sampler = FrameSampler(dataset)
        return sampler
    if is_distributed:
        from .samplers import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
    else:
        batch_sampler = cfg.test.batch_sampler

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        raise NotImplementedError

    if max_iter != -1:
        from .samplers import IterationBasedBatchSampler
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    import numpy as np
    import time
    # np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_collator(cfg, is_train):
    _collators = {
    }
    from torch.utils.data.dataloader import default_collate
    collator = cfg.train.collator if is_train else cfg.test.collator
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate

def Dataloader(cfg, split='train', is_train=True, start=0):
    is_distributed = cfg.distributed
    if split == 'train' and is_train:
        batch_size = cfg.train.batch_size
        max_iter = cfg.train.ep_iter
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False
        max_iter = -1
    if split == 'train' and is_train:
        dataset = load_object(cfg.data_train_module, cfg.data_train_args)
    elif split == 'train' and not is_train:
        cfg.data_train_args.split = 'test'
        dataset = load_object(cfg.data_train_module, cfg.data_train_args)
    elif split in ['test', 'val']:
        dataset = load_object(cfg.data_val_module, cfg.data_val_args)
    elif split == 'demo':
        dataset = load_object(cfg.data_demo_module, cfg.data_demo_args)
    elif split == 'mesh':
        dataset = load_object(cfg.data_mesh_module, cfg.data_mesh_args)
    else:
        raise NotImplementedError
    is_train = (split == 'train') and is_train
    sampler = make_data_sampler(cfg, dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
    num_workers = cfg.train.num_workers if is_train else cfg.test.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)

    return data_loader