import torch
import random
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .total_samples_sampler import TotalSamplesSampler


class SpeechTextDatset(Dataset):
    def __init__(
        self,
        data_list: str,
        tokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.uids = []
        self.paths = []
        self.lengths = []
        self.trans = []
        with open(data_list) as f:
            for line in f.readlines():
                line = line.strip()
                uid, path, samples, text = line.split("\t")
                self.uids.append(uid)
                self.paths.append(path)
                self.lengths.append(int(samples))
                self.trans.append(text)

    def __len__(self):
        return len(self.paths)

    def get_length(self, index):
        return self.lengths[index]

    def __getitem__(self, index: int):
        path = self.paths[index]
        wav, sr = torchaudio.load(path)
        assert sr == 16000
        wav = wav.reshape(-1)

        trans = self.trans[index]
        tokens = torch.LongTensor(
            self.tokenizer.encode(trans, add_special_tokens=False)
        )

        return wav, tokens

    @classmethod
    def get_dataloader(
        cls,
        data_list: str,
        tokenizer,
        total_samples: int = 10240000,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        dataset = cls(data_list, tokenizer)
        lengths = [dataset.get_length(index) for index in range(len(dataset))]
        batch_sampler = TotalSamplesSampler(lengths, total_samples, shuffle)

        def collate_fn(samples):
            wavs = []
            lengths = []
            all_tokens = []
            all_tokens_len = []
            for wav, tokens in samples:
                wavs.append(wav)
                lengths.append(len(wav))
                all_tokens.append(tokens)
                all_tokens_len.append(len(tokens))
            wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
            lengths = torch.LongTensor(lengths)
            all_tokens = pad_sequence(
                all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            all_tokens_len = torch.LongTensor(all_tokens_len)
            return wavs, lengths, all_tokens, all_tokens_len

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return dataloader
