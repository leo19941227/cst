import math

import torch
from torch.utils.data import Sampler, BatchSampler


class TotalSamplesSampler(BatchSampler):
    """A simple BatchSampler that yields a list of indices for each batch."""

    def __init__(self, lengths, total_samples, shuffle=False, seed=1227):
        self.lengths = lengths
        self.total_samples = total_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(list(iter(self)))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.lengths), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.lengths)))  # type: ignore[arg-type]

        def count_total_samples(batch_indices):
            max_length = max([self.lengths[index] for index in batch_indices])
            return max_length * len(batch_indices)

        batch_indices = []
        for index in indices:
            total_samples = count_total_samples(batch_indices + [index])
            if total_samples > self.total_samples:
                yield batch_indices
                batch_indices = [index]
            elif total_samples == self.total_samples:
                yield batch_indices + [index]
                batch_indices = []
            else:
                batch_indices.append(index)


class DistributedSamplerWrapper(Sampler):
    """
    Wrapper for a batch sampler to be used in distributed training.

    This sampler wraps an existing batch sampler, and splits its batches across
    the different distributed processes so that each process sees a different subset.

    Args:
        sampler (Sampler): Base batch sampler (should yield batches, e.g. lists of indices).
        num_replicas (int, optional): Number of processes participating in distributed training.
                                      If None, will be inferred from torch.distributed.
        rank (int, optional): Rank of the current process.
                              If None, will be inferred from torch.distributed.
        shuffle (bool, optional): Whether to shuffle the batches (default: True).
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        self.sampler = sampler

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Distributed training not available, please provide num_replicas"
                )
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Distributed training not available, please provide rank"
                )
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        # Number of batches in the base sampler
        self.num_batches = len(self.sampler)
        # Compute the number of batches per process (ceil to cover all batches)
        self.num_samples = int(math.ceil(self.num_batches * 1.0 / self.num_replicas))
        # Total number of batches after padding to be evenly divisible
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        # Get the list of batches from the base batch sampler.
        batches = list(self.sampler)

        # Pad with extra batches if necessary to make the number of batches evenly divisible.
        if len(batches) < self.total_size:
            padding = batches[: (self.total_size - len(batches))]
            batches += padding
        else:
            batches = batches[: self.total_size]

        # Subsample: Each process gets every num_replicas-th batch starting from its rank.
        distributed_batches = batches[self.rank : self.total_size : self.num_replicas]
        return iter(distributed_batches)

    def __len__(self):
        return self.num_samples
