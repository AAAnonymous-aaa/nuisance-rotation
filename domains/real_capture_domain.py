import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import collections
import io
import json

import pandas as pd
from PIL import Image


REPO = "Noaman/midv500"
SPLITS = ("train-00000-of-00002", "train-00001-of-00002",
          "validation-00000-of-00001")


def load_rows(split):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, f"data/{split}.parquet", repo_type="dataset")
    frame = pd.read_parquet(path, columns=["pixel_values"])
    for cell in frame["pixel_values"]:
        name = cell["path"] if isinstance(cell, dict) else cell
        blob = cell["bytes"] if isinstance(cell, dict) else None
        yield name, blob


def main():
    parser = argparse.ArgumentParser(description="Real second-capture domain.")
    parser.add_argument("--out", default="dataset/midv_capture")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--min-captures", type=int, default=3,
                        help="keep documents with at least this many captures")
    args = parser.parse_args()

    by_document = collections.defaultdict(list)
    for split in SPLITS:
        for name, blob in load_rows(split):
            if blob is None:
                continue
            document = name.split("_")[0]
            by_document[document].append((name, blob))
    kept = {d: rows for d, rows in by_document.items()
            if len(rows) >= args.min_captures}
    print(f"{sum(len(v) for v in by_document.values())} images, "
          f"{len(by_document)} documents, {len(kept)} with >= "
          f"{args.min_captures} captures")

    manifest = []
    for document in sorted(kept):
        folder = os.path.join(args.out, document)
        os.makedirs(folder, exist_ok=True)
        for index, (name, blob) in enumerate(sorted(kept[document], key=lambda r: r[0])):
            path = os.path.join(folder, f"c{index:02d}.jpg")
            if not os.path.exists(path):
                image = Image.open(io.BytesIO(blob)).convert("RGB")
                scale = args.max_side / max(image.size)
                if scale < 1:
                    image = image.resize(
                        (max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))),
                        Image.BICUBIC,
                    )
                image.save(path, quality=88)
            manifest.append(
                {
                    "topic": document[0],
                    "document": document,
                    "layout": index,
                    "path": path,
                    "source": name,
                }
            )
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    counts = [len(v) for v in kept.values()]
    print(f"real capture domain: {len(manifest)} images, {len(kept)} documents, "
          f"{min(counts)}-{max(counts)} captures each")


if __name__ == "__main__":
    main()
