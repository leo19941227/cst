class CharacterTokenizer:
    def __init__(self):
        self._vocab_list = ["<pad>"] + list(" 'abcdefghijklmnopqrstuvwxyz")
        self._vocab2idx = {v: idx for idx, v in enumerate(self._vocab_list)}

    @property
    def pad_token_id(self):
        return 0

    def __len__(self):
        return len(self._vocab_list)

    def encode(self, s, add_special_tokens=False):
        # Always strip trailing space, \r and \n
        s = s.strip("\r\n ").lower()
        # Manually append eos to the end
        return [self.vocab_to_idx(v) for v in s]

    def decode(self, idxs):
        vocabs = []
        for t, idx in enumerate(idxs):
            v = self.idx_to_vocab(idx)
            if idx == self.pad_token_id:
                continue
            else:
                vocabs.append(v)
        return "".join(vocabs)

    def vocab_to_idx(self, vocab):
        return self._vocab2idx[vocab]

    def idx_to_vocab(self, idx):
        return self._vocab_list[idx]
