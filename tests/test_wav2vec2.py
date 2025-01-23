from cst.model.wav2vec2 import Wav2Vec2Config
from cst.model.wav2vec2 import Wav2Vec2EncoderStableLayerNorm


def test_wav2vec2():
    config = Wav2Vec2Config()
    model = Wav2Vec2EncoderStableLayerNorm(config)
