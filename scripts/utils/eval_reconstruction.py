import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from s3prl.nn import S3PRLUpstream

from cst.datasets.audio import AudioDatset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_list")
    parser.add_argument("original_name")
    parser.add_argument("reduced_name")
    args = parser.parse_args()

    original = S3PRLUpstream(args.original_name).eval().cuda()
    reduced = S3PRLUpstream(args.reduced_name).eval().cuda()

    dataloader = AudioDatset.get_inference_dataloader(args.data_list, 8)

    with torch.no_grad():
        diffs = []
        for wavs, wavs_len in tqdm(dataloader):
            wavs = wavs.cuda()
            wavs_len = wavs_len.cuda()
            original_hs, original_hs_len = original(wavs, wavs_len)
            reduced_hs, reduced_hs_len = reduced(wavs, wavs_len)
            original_hs, original_hs_len = original_hs[-1], original_hs_len[-1]
            reduced_hs, reduced_hs_len = reduced_hs[-1], reduced_hs_len[-1]

            assert (reduced_hs_len.max() - original_hs_len.max()) <= 10
            original_hs = original_hs[:, : reduced_hs_len.max(), :]
            bsz, seqlen, size = original_hs.shape
            diff = (
                (
                    original_hs.reshape(bsz * seqlen, size)
                    - reduced_hs.reshape(bsz * seqlen, size)
                )
                .abs()
                .mean(dim=-1)
            )
            diffs.append(diff.detach().cpu())
        diffs = torch.cat(diffs, dim=0)
        avg = diffs.mean().item()
        print(avg)
