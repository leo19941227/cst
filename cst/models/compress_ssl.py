import math

import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from s3prl.nn import S3PRLUpstream

from cst.datasets.audio import AudioDatset
from cst.modules.wav2vec2 import CausalConv1d
from cst.modules.distribution import DiagonalGaussianDistribution
from cst.modules.autoencoder import SemanticAutoEncoder, SemanticAutoEncoderConfig


class SimpleModel(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.project1 = nn.Linear(hidden_size, hidden_size)
        self.project2 = nn.Linear(hidden_size, latent_size * 2)
        self.project3 = nn.Linear(latent_size, hidden_size)
        self.project4 = nn.Linear(hidden_size, hidden_size)

    def encode(self, hs, hs_len):
        moments = self.project2(F.relu(self.project1(hs)))
        posteriors = DiagonalGaussianDistribution(moments)
        return posteriors, hs_len

    def decode(self, latent, latent_len):
        hs = self.project4(F.relu(self.project3(latent)))
        return hs, latent_len


class CompressSSL(L.LightningModule):
    def __init__(
        self,
        upstream_name: str,
        latent_size: int,
        autoencoder_name: str = "transformers",
        autoencoder_conf=None,
        lr: float = 1.0e-4,
        kl_weight: float = 1.0,
        logvar_init: float = 0.0,
        initializer_range: float = 0.02,
        sample_posterior: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.initializer_range = initializer_range
        self.sample_posterior = sample_posterior

        self.upstream = S3PRLUpstream(upstream_name)
        self.upstream.requires_grad_(False)

        upstream_size = self.upstream.hidden_sizes[-1]
        if autoencoder_name == "transformers":
            autoencoder_conf["representation_size"] = upstream_size
            config = SemanticAutoEncoderConfig(**autoencoder_conf)
            self.autoencoder = SemanticAutoEncoder(config)
        elif autoencoder_name == "simple":
            self.autoencoder = SimpleModel(upstream_size, latent_size)
        else:
            raise ValueError(f"Unsupported autoencoder_name {autoencoder_name}")
        self.autoencoder.apply(self._init_weights)

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

    def encode(self, wavs, wavs_len):
        self.upstream.eval()
        with torch.no_grad():
            all_hs, all_hs_len = self.upstream(wavs, wavs_len)
            hs, hs_len = all_hs[-1], all_hs_len[-1]

        posteriors, latent_len = self.autoencoder.encode(hs, hs_len)
        return posteriors, latent_len

    def decode(self, latents, latents_len):
        dec, dec_len = self.autoencoder.decode(latents, latents_len)
        return dec, dec_len

    def forward(self, wavs, wavs_len):
        posteriors, latent_len = self.encode(wavs, wavs_len)
        if self.sample_posterior:
            latents = posteriors.sample()
        else:
            latents = posteriors.mode()
        dec, dec_len = self.decode(latents, latents_len)

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
        rec_loss = (
            (rec_loss * valid_mask.unsqueeze(-1)).sum()
            / dec_len.sum().item()
            / rec_loss.size(-1)
        )
        return total_loss, nll_loss, kl_loss, rec_loss

    def training_step(self, batch, batch_idx):
        wavs, wavs_len = batch
        total_loss, nll_loss, kl_loss, rec_loss = self(wavs, wavs_len)
        self.log_dict(
            {
                "train/loss": total_loss,
                "train/nll_loss": nll_loss,
                "train/kl_loss": kl_loss,
                "train/rec_loss": rec_loss,
            },
            prog_bar=True,
            on_step=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        wavs, wavs_len = batch
        total_loss, nll_loss, kl_loss, rec_loss = self(wavs, wavs_len)
        self.log_dict(
            {
                "valid/loss": total_loss,
                "valid/nll_loss": nll_loss,
                "valid/kl_loss": kl_loss,
                "valid/rec_loss": rec_loss,
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
