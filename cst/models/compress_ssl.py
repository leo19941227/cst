import math

import torch
import torch.nn as nn
import lightning as L
from s3prl.nn import S3PRLUpstream

from cst.datasets.audio import AudioDatset
from cst.modules.wav2vec2 import CausalConv1d
from cst.modules.autoencoder import SemanticAutoEncoder, SemanticAutoEncoderConfig


class CompressSSL(L.LightningModule):
    def __init__(
        self,
        upstream_name: str,
        autoencoder_conf: SemanticAutoEncoderConfig,
        lr: float,
        kl_weight: float = 1.0,
        logvar_init: float = 0.0,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.upstream = S3PRLUpstream(upstream_name)
        self.upstream.requires_grad_(False)

        config = SemanticAutoEncoderConfig(**autoencoder_conf)
        self.autoencoder = SemanticAutoEncoder(config)

        self.lr = lr
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.initializer_range = initializer_range

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, CausalConv1d):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2
                * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
                nn.init.uniform_(module.bias, a=-k, b=k)

    def forward(self, wavs, wavs_len):
        self.upstream.eval()
        with torch.no_grad():
            all_hs, all_hs_len = self.upstream(wavs, wavs_len)
            hs, hs_len = all_hs[-1], all_hs_len[-1]
        posteriors, latent_len, dec, dec_len = self.autoencoder(hs, hs_len)

        hs = hs[:, : dec_len.max()].contiguous()
        dec = dec[:, : dec_len.max()].contiguous()
        valid_mask = torch.lt(
            torch.arange(dec_len.max()).unsqueeze(0).to(hs.device), dec_len.unsqueeze(1)
        )
        rec_loss = torch.abs(hs - dec)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar + math.log(2.0)
        nll_loss = nll_loss * valid_mask.unsqueeze(-1)
        nll_loss = torch.sum(nll_loss) / dec_len.sum().item() / nll_loss.size(-1)

        kl_loss = posteriors.kl_per_dim()
        valid_mask = torch.lt(
            torch.arange(kl_loss.size(1)).unsqueeze(0).to(hs.device),
            latent_len.unsqueeze(1),
        )
        kl_loss = kl_loss * valid_mask
        kl_loss = torch.sum(kl_loss) / latent_len.sum().item()

        total_loss = nll_loss + self.kl_weight * kl_loss
        return total_loss, nll_loss, kl_loss

    def training_step(self, batch, batch_idx):
        wavs, wavs_len = batch
        total_loss, nll_loss, kl_loss = self(wavs, wavs_len)
        self.log_dict(
            {
                "train/loss": total_loss,
                "train/nll_loss": nll_loss,
                "train/kl_loss": kl_loss,
            },
            prog_bar=True,
            on_step=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        wavs, wavs_len = batch
        total_loss, nll_loss, kl_loss = self(wavs, wavs_len)
        self.log_dict(
            {
                "valid/loss": total_loss,
                "valid/nll_loss": nll_loss,
                "valid/kl_loss": kl_loss,
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        return total_loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.lr)

    def setup_data(
        self,
        train_conf: dict,
        valid_conf: dict,
    ):
        self.train_dl = AudioDatset.get_dataloader(**train_conf)
        self.valid_dl = AudioDatset.get_dataloader(**valid_conf)

    def get_dataloader(self, split: str):
        if split == "train":
            return self.train_dl
        elif split == "valid":
            return self.valid_dl
