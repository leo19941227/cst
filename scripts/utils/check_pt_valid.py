import argparse
from pathlib import Path

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_list")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    with open(args.data_list) as f:
        files = [line.strip() for line in f.readlines()]
        stems = [Path(file).stem for file in files]
        pts = [Path(args.output_dir) / f"{stem}.pt" for stem in stems]
        success = True
        for pt in pts:
            try:
                torch.load(pt)
            except:
                success = False

    if success:
        exit(0)
    else:
        exit(1)
