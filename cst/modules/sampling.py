import torch
import torch.nn as nn

from .wav2vec2 import CausalConv1d


class LinearDownsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): batch_size x seqlen x hidden_size
        """
        bsz, seqlen, hs = hidden_states.shape
        hidden_states = hidden_states[:, : seqlen // 2 * 2].reshape(
            bsz, seqlen // 2, hs * 2
        )
        hidden_states = self.project(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class LinearUpsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): batch_size x seqlen x hidden_size
        """
        bsz, seqlen, hs = hidden_states.shape
        projected = self.project(hidden_states)
        hidden_states = projected.reshape(bsz, 2 * seqlen, self.config.hidden_size)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class ConvUpsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = CausalConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=1
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x


class ConvDownsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = CausalConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        xd = self.conv(x)
        xd = xd.transpose(1, 2)
        return xd


class InterpolateUpsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = x.transpose(1, 2)
        return x


class AverageDownsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.transpose(1, 2)
        return x
