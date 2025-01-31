import argparse
from pathlib import Path


def find_text(audio_path):
    audio_path = Path(audio_path)
    text_filename = "-".join(audio_path.stem.split("-")[:2])
    text_filename = f"{text_filename}.trans.txt"
    with (audio_path.parent / text_filename).open() as f:
        for line in f.readlines():
            line = line.strip()
            fileid, trans = line.split(" ", maxsplit=1)
            if fileid == audio_path.stem:
                return trans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_list")
    args = parser.parse_args()

    with open(args.data_list) as f:
        for line in f.readlines():
            line = line.strip()
            path, num_frames = line.split("\t")
            trans = find_text(path)
            print(f"{Path(path).stem}\t{path}\t{num_frames}\t{trans}")
