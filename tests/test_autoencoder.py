import torch
from cst.model.autoencoder import SemanticAutoEncoder, SemanticAutoEncoderConfig

def test_autoencoder():
    config = SemanticAutoEncoderConfig()
    model = SemanticAutoEncoder(config)
    representation = torch.randn(32, 128, config.representation_size)
    reconstructed, posterior = model(representation)
    assert reconstructed.shape == (32, 128, config.representation_size)
    assert posterior.sample().squeeze(-1).transpose(1, 2).shape == (32, int(128 / 8), config.latent_size)
