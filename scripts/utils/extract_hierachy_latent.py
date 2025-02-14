import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from cst.datasets.audio import AudioDatset
from cst.models.hierarchical_vae import HierarchicalVae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt1")
    parser.add_argument("ckpt2")
    parser.add_argument("stack", type=int)
    parser.add_argument("data_list")
    parser.add_argument("output_dir")
    parser.add_argument("--chunk_size", type=int, default=320000)
    args = parser.parse_args()

    model = HierarchicalVae(args.ckpt1, args.ckpt2, args.stack).eval().cuda()

    dataloader = AudioDatset.get_inference_dataloader(args.data_list, 8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for wavs, wavs_len, stems in tqdm(dataloader):
            wav = wavs[0].cuda()
            latent_segs = []
            for start in range(0, len(wav), args.chunk_size):
                end = min(start + args.chunk_size, len(wav))
                wav_seg = wav[start:end].unsqueeze(0)
                wav_seg_len = torch.LongTensor([wav_seg.size(1)]).cuda()

                hs, hs_len, latent1, latent1_len, latent2, latent2_len = model.encode(
                    wav_seg, wav_seg_len
                )
                latent_seg = latent2[0].detach().cpu()
                latent_segs.append(latent_seg)
            latent = torch.cat(latent_segs, dim=0)
            torch.save(latent, str(output_dir / f"{stems[0]}.pt"))
