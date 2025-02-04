import torch
from torch.utils.data import BatchSampler


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
