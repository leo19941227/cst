import torch
import torch.nn as nn

from cst.models.compress_ssl import CompressSSL


class HierarchicalVae(nn.Module):
    def __init__(self, vae1_path: str, vae2_path: str, stack: int = 1):
        super().__init__()
        self.vae1 = CompressSSL.load_from_checkpoint(vae1_path, map_location="cpu")
        self.vae2 = CompressSSL.load_from_checkpoint(vae2_path, map_location="cpu")
        self.stack = stack

    def encode(self, wavs, wavs_len):
        hs, hs_len, posteriors, latent1_len = self.vae1.encode(wavs, wavs_len)
        latent1 = posteriors.mode()

        bsz, seqlen, size = latent1.shape
        latent1_stack = latent1[:, : seqlen // self.stack * self.stack, :].reshape(
            bsz, seqlen // self.stack, size * self.stack
        )
        latent1_stack_len = torch.div(latent1_len, self.stack, rounding_mode="floor")

        posteriors, latent2_len = self.vae2.encode_representation(latent1_stack, latent1_stack_len)
        latent2 = posteriors.mode()
        return hs, hs_len, latent1, latent1_len, latent2, latent2_len

    def decode_latent2(self, latent2, latent2_len):
        dec2, dec2_len = self.vae2.decode(latent2, latent2_len)

        if self.stack > 1:
            bsz, seqlen, size = dec2.shape
            dec2 = dec2.reshape(bsz, seqlen * self.stack, size // self.stack)
            dec2_len = dec2_len * self.stack

        dec1, dec1_len = self.vae1.decode(dec2, dec2_len)
        return dec1, dec1_len, dec2, dec2_len


    def decode_latent1(self, latent1, latent1_len):
        dec2, dec2_len = latent1, latent1_len

        if self.stack > 1:
            bsz, seqlen, size = dec2.shape
            dec2 = dec2.reshape(bsz, seqlen * self.stack, size // self.stack)
            dec2_len = dec2_len * self.stack

        dec1, dec1_len = self.vae1.decode(dec2, dec2_len)
        return dec1, dec1_len
