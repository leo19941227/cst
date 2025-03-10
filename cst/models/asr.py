import logging

import torch
import torch.nn as nn
import lightning as L
import torch.nn.init as init
import torch.nn.functional as F
from s3prl.nn import S3PRLUpstream
from transformers import AutoTokenizer

from cst.modules.rnn import RNNs
from cst.datasets.specaug import SpecAug
from cst.datasets.tokenizer import CharacterTokenizer
from cst.datasets.speech_text import SpeechTextDatset

import editdistance as ed

logger = logging.getLogger(__name__)


def cer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


def wer(hypothesis, groundtruth):
    err = 0
    tot = 0
    for p, t in zip(hypothesis, groundtruth):
        p = p.split(" ")
        t = t.split(" ")
        err += float(ed.eval(p, t))
        tot += len(t)
    return err / tot


class SimpleModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.project1 = nn.Linear(hidden_size, hidden_size)
        self.project2 = nn.Linear(hidden_size, output_size)

    def forward(self, hs, hs_len):
        hs = self.project2(F.relu(self.project1(hs)))
        return hs, hs_len


def init_weights(m):
    # Initialize Linear layers with Xavier initialization
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    # Initialize LSTM layers
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                # Xavier initialization for input-to-hidden weights.
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                # Orthogonal initialization for hidden-to-hidden (recurrent) weights.
                init.orthogonal_(param.data)
            elif "bias" in name:
                # Zero-initialize biases.
                param.data.fill_(0)
                # Optionally, set forget gate bias to 1 if this is an LSTM.
                # PyTorch LSTM biases are arranged as: [input, forget, cell, output].
                # This block checks if the bias shape corresponds to an LSTM's bias.
                hidden_size = m.hidden_size
                if param.data.shape[0] == 4 * hidden_size:
                    start, end = (
                        hidden_size,
                        2 * hidden_size,
                    )  # Forget gate bias indices.
                    param.data[start:end].fill_(1.0)


class CtcASR(L.LightningModule):
    def __init__(
        self,
        upstream_name: str,
        project_dim: int = 512,
        tokenizer_name: str = "bert-base-uncased",
        specaug_conf: dict = None,
        downstream_name: str = "rnn",
        downstream_conf: dict = None,
        upstream_conf: dict = None,
        lr: float = 1.0e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if tokenizer_name == "char":
            self.tokenizer = CharacterTokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        upstream_conf = upstream_conf or {}
        self.upstream = S3PRLUpstream(upstream_name, extra_conf=upstream_conf)
        self.upstream.requires_grad_(False)
        if (
            hasattr(self.upstream, "embeddings")
            and self.upstream.embeddings is not None
        ):
            self.upstream.embeddings.requires_grad_(True)

        # specaug
        self.specaug = None
        if specaug_conf is not None:
            logger.info("Apply specaug")
            self.specaug = SpecAug(**specaug_conf)

        self.projector = nn.Linear(self.upstream.hidden_sizes[-1], project_dim)
        output_size = len(self.tokenizer)
        if downstream_name == "rnn":
            self.model = RNNs(
                project_dim,
                output_size,
                **downstream_conf,
            )
        elif downstream_name == "probing":
            self.model = SimpleModel(project_dim, output_size)
        else:
            raise ValueError(f"Unsupported downstream_name: {downstream_name}")

        self.objective = nn.CTCLoss(
            blank=self.tokenizer.pad_token_id,
            zero_infinity=True,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        self.projector.apply(init_weights)
        self.model.apply(init_weights)

    def forward(self, wavs, wavs_len, tokens, tokens_len):
        all_hs, all_hs_len = self.upstream(wavs, wavs_len)
        hs, hs_len = all_hs[-1], all_hs_len[-1]

        if self.specaug is not None and self.training:
            hs, _ = self.specaug(hs)

        hs = self.projector(hs)
        logits, log_probs_len = self.model(hs, hs_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = self.objective(
            log_probs.transpose(0, 1),  # (N, T, C) -> (T, N, C)
            tokens,
            log_probs_len,
            tokens_len,
        )

        pred_tokens = log_probs.argmax(dim=-1)  # greedy decoding
        filtered_tokens = []
        for pred_token in pred_tokens:
            pred_token = pred_token.unique_consecutive()
            filtered_token = [
                token
                for token in pred_token.tolist()
                if token != self.tokenizer.pad_token_id
            ]
            filtered_tokens.append(filtered_token)

        hyps = [self.tokenizer.decode(h) for h in filtered_tokens]
        refs = [
            self.tokenizer.decode(
                [t for t in g.tolist() if t != self.tokenizer.pad_token_id]
            )
            for g in tokens
        ]

        return loss, hyps, refs

    def training_step(self, batch, batch_id):
        wavs, wavs_len, tokens, tokens_len = batch
        loss, hyps, refs = self(wavs, wavs_len, tokens, tokens_len)

        self.log_dict(
            {
                "train/loss": loss,
            },
            prog_bar=True,
            on_step=True,
            logger=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.hyps = []
        self.refs = []

    def validation_step(self, batch, batch_id):
        wavs, wavs_len, tokens, tokens_len = batch
        loss, hyps, refs = self(wavs, wavs_len, tokens, tokens_len)
        self.hyps.extend(hyps)
        self.refs.extend(refs)

        self.log_dict(
            {
                "train/loss": loss,
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "valid/cer": cer(self.hyps, self.refs),
                "valid/wer": wer(self.hyps, self.refs),
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.lr)

    def setup_data(
        self,
        train_conf: dict,
        valid_conf: dict,
    ):
        self.train_dl = SpeechTextDatset.get_dataloader(
            **train_conf, tokenizer=self.tokenizer
        )
        self.valid_dl = SpeechTextDatset.get_dataloader(
            **valid_conf, tokenizer=self.tokenizer
        )

    def get_dataloader(self, split: str):
        if split == "train":
            return self.train_dl
        elif split == "valid":
            return self.valid_dl
