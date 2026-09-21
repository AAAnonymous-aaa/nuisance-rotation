import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import io
import json

import numpy as np
from PIL import Image


REPO = "pixparse/docvqa-single-page-questions"
SHARDS = [f"data/train-{index:05d}-of-00036.parquet" for index in
          (2, 5, 20, 35, 32, 1, 3, 4)]


def load_images(shards):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    seen = {}
    for shard in shards:
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        frame = pd.read_parquet(path, columns=["image"])
        for cell in frame["image"]:
            name = cell["path"]
            if name not in seen:
                seen[name] = cell["bytes"]
        print(f"  {shard}: {len(seen)} unique documents so far")
    return seen


def jpeg_roundtrip(image, quality=30):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def rescan(image, scale=0.5):
    small = image.resize((max(1, int(image.width * scale)),
                          max(1, int(image.height * scale))), Image.BICUBIC)
    return small.resize(image.size, Image.BICUBIC)


def skew_with_illumination(image, degrees=0.6, border=10):
    rotated = image.rotate(degrees, resample=Image.BICUBIC, expand=False,
                           fillcolor=(255, 255, 255))
    array = np.asarray(rotated).astype(np.float32)
    ramp = np.linspace(0.85, 1.12, array.shape[1], dtype=np.float32)[None, :, None]
    array = np.clip(array * ramp, 0, 255)
    array = np.pad(array, ((border, border), (border, border), (0, 0)),
                   mode="constant", constant_values=255)
    return Image.fromarray(array.astype(np.uint8))


PASSES = (lambda image: image, jpeg_roundtrip, rescan, skew_with_illumination)


def main():
    parser = argparse.ArgumentParser(description="Large real-scan domain.")
    parser.add_argument("--out", default="dataset/docvqa_scan")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=800)
    args = parser.parse_args()

    images = load_images(SHARDS[: args.shards])
    print(f"{len(images)} unique real scanned documents")
    manifest = []
    for index, (name, blob) in enumerate(sorted(images.items())):
        base = Image.open(io.BytesIO(blob)).convert("RGB")
        scale = args.max_side / max(base.size)
        if scale < 1:
            base = base.resize((max(1, round(base.width * scale)),
                                max(1, round(base.height * scale))), Image.BICUBIC)
        folder = os.path.join(args.out, f"d{index:05d}")
        os.makedirs(folder, exist_ok=True)
        for layout, transform in enumerate(PASSES):
            path = os.path.join(folder, f"l{layout:02d}.png")
            if not os.path.exists(path):
                transform(base).save(path)
            manifest.append({
                "topic": name[:4],
                "document": f"docvqa::{index:05d}",
                "layout": layout,
                "path": path,
                "source": name,
            })
        if (index + 1) % 200 == 0:
            print(f"  rendered {index + 1}/{len(images)}")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"large real-scan domain: {len(manifest)} images, "
          f"{len({r['document'] for r in manifest})} documents, "
          f"{len({r['layout'] for r in manifest})} passes")


if __name__ == "__main__":
    main()
