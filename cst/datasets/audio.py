import torch
import random
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler, DataLoader


class TotalSamplesSampler(Sampler[list]):
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


class AudioDatset(Dataset):
    def __init__(
        self,
        data_list: str,
        training: bool,
        max_samples: int = 320000,
        min_samples: int = 48000,
    ):
        super().__init__()
        self.training = training
        self.max_samples = max_samples
        self.min_samples = min_samples

        self.paths = []
        self.lengths = []
        with open(data_list) as f:
            for line in f.readlines():
                line = line.strip()
                path, samples = line.split("\t", maxsplit=1)
                samples = int(samples)
                length = min(max_samples, samples)
                if training and length < min_samples:
                    continue
                self.paths.append(path)
                self.lengths.append(length)

    def __len__(self):
        return len(self.paths)

    def get_length(self, index):
        return self.lengths[index]

    def __getitem__(self, index: int):
        path = self.paths[index]
        wav, sr = torchaudio.load(path)
        assert sr == 16000
        wav = wav.reshape(-1)
        if self.training and len(wav) > self.max_samples:
            start = random.randint(0, len(wav) - self.max_samples - 1)
            wav = wav[start : start + self.max_samples]
        return wav

    @classmethod
    def get_dataloader(
        cls,
        data_list: str,
        training: bool,
        max_samples: int = 320000,
        min_samples: int = 48000,
        total_samples: int = 10240000,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        dataset = cls(data_list, training, max_samples, min_samples)
        lengths = [dataset.get_length(index) for index in range(len(dataset))]
        batch_sampler = TotalSamplesSampler(lengths, total_samples, shuffle)

        def collate_fn(samples):
            wavs = []
            lengths = []
            for wav in samples:
                wavs.append(wav)
                lengths.append(len(wav))
            wavs = pad_sequence(wavs, batch_first=True)
            lengths = torch.LongTensor(lengths)
            return wavs, lengths

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return dataloader
