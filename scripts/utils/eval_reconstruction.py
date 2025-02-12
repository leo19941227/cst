import argparse

import torch
from tqdm import tqdm

from cst.datasets.audio import AudioDatset
from cst.models.compress_ssl import CompressSSL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_list")
    parser.add_argument("ckpt")
    args = parser.parse_args()

    model = (
        CompressSSL.load_from_checkpoint(args.ckpt, map_location="cpu").eval().cuda()
    )

    dataloader = AudioDatset.get_inference_dataloader(args.data_list, 8)

    with torch.no_grad():
        diffs = []
        for wavs, wavs_len in tqdm(dataloader):
            wavs = wavs.cuda()
            wavs_len = wavs_len.cuda()

            hs, hs_len, posteriors, latent_len = model.encode(wavs, wavs_len)
            latent = posteriors.mode()
            dec, dec_len = model.decode(latent, latent_len)

            assert abs(dec_len.max() - hs_len.max()) <= 10
            hs = hs[:, : dec_len.max(), :]
            bsz, seqlen, size = hs.shape
            diff = (
                (hs.reshape(bsz * seqlen, size) - dec.reshape(bsz * seqlen, size))
                .abs()
                .mean(dim=-1)
            )
            diffs.append(diff.detach().cpu())
        diffs = torch.cat(diffs, dim=0)
        avg = diffs.mean().item()
        print(avg)
