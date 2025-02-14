import argparse
from pathlib import Path

import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_dir")
    parser.add_argument("output_file")
    args = parser.parse_args()

    with open(args.output_file, "w") as f:

        for path in tqdm(list(Path(args.pt_dir).glob("*.pt"))):
            path = path.absolute()
            try:
                hs = torch.load(path)
                seqlen = hs.size(0)
                print(f"{path}\t{seqlen}", file=f)
            except:
                print(f"fail to load {path}")
