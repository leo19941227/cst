from pathlib import Path

import torch
import random
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .total_samples_sampler import TotalSamplesSampler


class AudioDatset(Dataset):
    def __init__(
        self,
        data_list: str,
        training: bool,
        max_samples: int = 320000,
        min_samples: int = 48000,
        return_stem: bool = False,
    ):
        super().__init__()
        self.training = training
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.return_stem = return_stem

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
        if self.return_stem:
            return wav, Path(path).stem
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

    @classmethod
    def get_inference_dataloader(
        cls,
        data_list: str,
        num_workers: int = 0,
    ):
        dataset = cls(data_list, False, return_stem=True)

        def collate_fn(samples):
            wavs = []
            lengths = []
            stems = []
            for wav, stem in samples:
                wavs.append(wav)
                lengths.append(len(wav))
                stems.append(stem)
            wavs = pad_sequence(wavs, batch_first=True)
            lengths = torch.LongTensor(lengths)
            return wavs, lengths, stems

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return dataloader
