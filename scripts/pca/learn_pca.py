import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def load_feature_shard(feat_dir, nshard, rank, percent):
    feat_path = f"{feat_dir}/{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i] : offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [load_feature_shard(feat_dir, nshard, r, percent) for r in range(nshard)],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_pca(
    feat_dir,
    nshard,
    pca_path,
    n_components,
    seed,
    percent,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, nshard, seed, percent)
    pca_model = PCA(n_components=n_components)
    pca_model.fit(feat)
    joblib.dump(pca_model, pca_path)
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("pca_path", type=str)
    parser.add_argument("n_components", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    args = parser.parse_args()
    learn_pca(**vars(args))
