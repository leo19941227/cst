import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from cst.datasets.audio import AudioDatset
from cst.models.compress_ssl import CompressSSL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")  # /livingrooms/public/leo19941227/vae_lr5e-4_spin_stack2_4transformer_latent8_kl1e-4.pt
    parser.add_argument("data_list")
    parser.add_argument("output_dir")
    parser.add_argument("--chunk_size", type=int, default=320000)
    args = parser.parse_args()

    model = (
        CompressSSL.load_from_checkpoint(args.ckpt, map_location="cpu").eval().cuda()
    )

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

                hs, hs_len, posteriors, latent_len = model.encode(wav_seg, wav_seg_len)
                latent_seg = posteriors.parameters[0].detach().cpu()
                latent_segs.append(latent_seg)
            latent = torch.cat(latent_segs, dim=0)
            torch.save(latent, str(output_dir / f"{stems[0]}.pt"))
