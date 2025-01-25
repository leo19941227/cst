import torch
import random
from cst.modules.autoencoder import SemanticAutoEncoder, SemanticAutoEncoderConfig

def test_autoencoder():
    config = SemanticAutoEncoderConfig()
    model = SemanticAutoEncoder(config)
    representation = torch.randn(32, 128, config.representation_size)
    length = torch.LongTensor([random.randint(0, 127) for _ in range(32)])
    posteriors, latent_len, dec, dec_len = model(representation, length)
    assert dec.shape == (32, 128, config.representation_size)
    assert posteriors.sample().shape == (32, int(128 / 8), config.latent_size)
    assert length.min() // 2 // 2 // 2 * 2 * 2 * 2 == dec_len.min()
    assert dec_len.dim() == 1
    assert dec_len.size(0) == dec.size(0)
