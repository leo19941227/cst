import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from cst.datasets.audio import AudioDatset
from cst.models.hierarchical_vae import HierarchicalVae


def compute_diff(original, original_len, reconstructed, reconstructed_len):
    assert abs(original_len.max() - reconstructed_len.max()) <= 5
    original = original[:, : reconstructed_len.max(), :]
    bsz, seqlen, size = original.shape
    diff = (
        (
            original.reshape(bsz * seqlen, size)
            - reconstructed.reshape(bsz * seqlen, size)
        )
        .abs()
        .mean(dim=-1)
    )
    return diff.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt1")
    parser.add_argument("ckpt2")
    parser.add_argument("stack", type=int)
    parser.add_argument("data_list")
    args = parser.parse_args()

    model = HierarchicalVae(args.ckpt1, args.ckpt2, args.stack).eval().cuda()

    dataloader = AudioDatset.get_inference_dataloader(args.data_list, 8)

    hs_diffs = []
    latent1_diffs = []
    with torch.no_grad():
        for wavs, wavs_len in tqdm(dataloader):
            wavs = wavs.cuda()
            wavs_len = wavs_len.cuda()

            hs, hs_len, latent1, latent1_len, latent2, latent2_len = model.encode(
                wavs, wavs_len
            )
            dec1, dec1_len, dec2, dec2_len = model.decode_latent2(latent2, latent2_len)

            latent1_diffs.append(compute_diff(latent1, latent1_len, dec2, dec2_len))
            hs_diffs.append(compute_diff(hs, hs_len, dec1, dec1_len))

        hs_diff = torch.cat(hs_diffs, dim=0)
        avg = hs_diff.mean().item()
        print(f"hs_diff: {avg}")

        latent1_diff = torch.cat(latent1_diffs, dim=0)
        avg = latent1_diff.mean().item()
        print(f"latent1_diff: {avg}")
