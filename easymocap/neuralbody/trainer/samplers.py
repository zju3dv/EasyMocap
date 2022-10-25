# copy from neuralbody
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
import math
import torch.distributed as dist


class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, sampler_meta):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.strategy = sampler_meta.strategy
        self.hmin, self.wmin = sampler_meta.min_hw
        self.hmax, self.wmax = sampler_meta.max_hw
        self.divisor = 32

    def generate_height_width(self):
        if self.strategy == 'origin':
            return -1, -1
        h = np.random.randint(self.hmin, self.hmax + 1)
        w = np.random.randint(self.wmin, self.wmax + 1)
        h = (h | (self.divisor - 1)) + 1
        w = (w | (self.divisor - 1)) + 1
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.sampler = self.batch_sampler.sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class FrameSampler(Sampler):
    """Sampler certain frames for test
    """

    def __init__(self, dataset):
        inds = np.arange(0, len(dataset.ims))
        ni = len(dataset.ims) // dataset.num_cams
        inds = inds.reshape(ni, -1)[::30]
        self.inds = inds.ravel()

    def __iter__(self):
        return iter(self.inds)

    def __len__(self):
        return len(self.inds)
