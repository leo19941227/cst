from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wav2vec2 import (
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2EncoderLayerStableLayerNorm,
)
from .distribution import DiagonalGaussianDistribution


class SemanticAutoEncoderConfig:
    r"""
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        positional_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D positional conv layers of the positional
            embedding. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
    """

    model_type = "semantic_autoencoder"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        layer_norm_eps=1e-5,
        positional_activation="gelu",
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        encoding_layer_sizes=None,
        decoding_layer_sizes=None,
        representation_size=1024,
        latent_size=8,
        sample_posterior=True,
    ):
        self.hidden_size = hidden_size
        self.positional_activation = positional_activation
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_eps = layer_norm_eps
        self.encoding_layer_sizes = (
            encoding_layer_sizes or [768, -2, 768, -2, 768, -2] + [768] * 4
        )
        self.decoding_layer_sizes = decoding_layer_sizes or [768] * 4 + [
            -3,
            768,
            -3,
            768,
            -3,
            768,
        ]
        self.representation_size = representation_size
        self.latent_size = latent_size
        self.sample_posterior = sample_posterior


class Downsample(nn.Module):
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


class Upsample(nn.Module):
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


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states,
    ):
        """
        Args:
            hidden_states: batch_size x seqlen x hidden_size
        """
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = Wav2Vec2EncoderLayerStableLayerNorm(config)
        # create a causal mask
        attn_mask = torch.ones(2000, 2000)
        attn_mask = torch.tril(attn_mask)
        attn_mask = 1 - attn_mask
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, hs):
        attn_mask = self.attn_mask[: hs.size(1), : hs.size(1)].to(dtype=hs.dtype)
        attn_mask = attn_mask * torch.finfo(hs.dtype).min
        attn_mask = (
            attn_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(hs.size(0), 1, hs.size(1), hs.size(1))
        )
        hidden_states = self.layer(hs, attention_mask=attn_mask)[0]
        return hidden_states


def new_config(config: SemanticAutoEncoderConfig, hidden_size: int):
    config = deepcopy(config)
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 3
    return config


class SemanticAutoEncoder(nn.Module):
    def __init__(self, config: SemanticAutoEncoderConfig):
        super().__init__()
        self.config = config

        self.pre_project = nn.Linear(config.representation_size, config.hidden_size)

        encoding_layers = []
        latest_size = config.hidden_size
        latest_config = config
        self.downsample_times = 0
        for layer_size in config.encoding_layer_sizes:
            if layer_size > 0:
                latest_config = new_config(config, layer_size)
                encoding_layers.extend(
                    [
                        nn.Linear(latest_size, layer_size),
                        TransformerLayer(latest_config),
                    ]
                )
                latest_size = layer_size
            elif layer_size == -1:
                encoding_layers.append(PositionalEmbedding(latest_config))
            elif layer_size == -2:
                encoding_layers.append(Downsample(latest_config))
                self.downsample_times += 1
            else:
                raise ValueError
        self.encoding_layers = nn.Sequential(*encoding_layers)

        moments_size = config.latent_size * 2
        self.encode_latent = nn.Linear(latest_size, moments_size)
        latest_size = config.latent_size
        latest_config = new_config(config, latest_size)

        decoding_layers = []
        for layer_size in config.decoding_layer_sizes:
            if layer_size > 0:
                latest_config = new_config(config, layer_size)
                decoding_layers.extend(
                    [
                        nn.Linear(latest_size, layer_size),
                        TransformerLayer(latest_config),
                    ]
                )
                latest_size = layer_size
            elif layer_size == -1:
                decoding_layers.append(PositionalEmbedding(latest_config))
            elif layer_size == -3:
                decoding_layers.append(Upsample(latest_config))
            else:
                raise ValueError
        self.decoding_layers = nn.Sequential(*decoding_layers)

        self.post_project = nn.Linear(config.hidden_size, config.representation_size)

    def encode(self, hs, hs_len):
        hs = self.pre_project(hs)
        hs = self.encoding_layers(hs)
        moments = self.encode_latent(hs)  # (bsz, seqlen, hs)
        moments = F.layer_norm(moments, moments.shape[-1:])
        posteriors = DiagonalGaussianDistribution(moments)

        for _ in range(self.downsample_times):
            hs_len = torch.div(hs_len, 2, rounding_mode="floor")

        return posteriors, hs_len

    def decode(self, latent, hs_len):
        hs = self.decoding_layers(latent)
        hs = self.post_project(hs)
        hs_len = hs_len * (2**self.downsample_times)
        return hs, hs_len

    def forward(self, hs, hs_len):
        posteriors, latent_len = self.encode(hs, hs_len)
        if self.config.sample_posterior:
            latent = posteriors.sample()
        else:
            latent = posteriors.mode()
        dec, dec_len = self.decode(latent, latent_len)
        return posteriors, latent_len, dec, dec_len
