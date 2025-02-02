# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import soundfile as sf
import torch
import torch.nn.functional as F
from s3prl.nn import S3PRLUpstream

from cst.feature_utils import get_path_iterator, dump_feature


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class S3PRLFeatureReader(object):
    def __init__(self, upstream_name, layer, max_chunk=1600000):
        self.model = S3PRLUpstream(upstream_name).cuda()
        self.model.eval()
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == 16000
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                lengths = torch.LongTensor([x_chunk.size(1)])
                feat_chunk, _ = self.model(x_chunk, lengths)
                feat_chunk = feat_chunk[self.layer]
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def main(audio_list, upstream_name, layer, nshard, rank, feat_dir, max_chunk):
    reader = S3PRLFeatureReader(upstream_name, layer, max_chunk)
    generator, num = get_path_iterator(audio_list, nshard, rank)
    dump_feature(reader, generator, num, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_list")
    parser.add_argument("upstream_name")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
