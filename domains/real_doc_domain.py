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


SPLITS = ("train", "test")
JPEG_QUALITY = 30
DOWNSCALE = 0.5
SKEW_DEGREES = 0.6
BORDER = 10


def load_funsd_images():

    import pandas as pd
    from huggingface_hub import hf_hub_download

    images = []
    for split in SPLITS:
        path = hf_hub_download(
            "nielsr/funsd",
            f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(path, columns=["image"])
        for index, row in enumerate(frame["image"]):
            blob = row["bytes"] if isinstance(row, dict) else row
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            images.append((f"{split}-{index:04d}", image))
    return images


def jpeg_roundtrip(image, quality=JPEG_QUALITY):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def rescan(image, scale=DOWNSCALE):
    small = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.BICUBIC,
    )
    return small.resize(image.size, Image.BICUBIC)


def skew_with_illumination(image, degrees=SKEW_DEGREES, border=BORDER):
    rotated = image.rotate(degrees, resample=Image.BICUBIC, expand=False,
                           fillcolor=(255, 255, 255))
    array = np.asarray(rotated).astype(np.float32)
    height, width = array.shape[:2]
    ramp = np.linspace(0.85, 1.12, width, dtype=np.float32)[None, :, None]
    array = np.clip(array * ramp, 0, 255)
    array = np.pad(array, ((border, border), (border, border), (0, 0)),
                   mode="constant", constant_values=255)
    return Image.fromarray(array.astype(np.uint8))


PASSES = (lambda image: image, jpeg_roundtrip, rescan, skew_with_illumination)


def main():
    parser = argparse.ArgumentParser(description="Real scanned-form domain.")
    parser.add_argument("--out", default="dataset/funsd_scan")
    parser.add_argument("--max-documents", type=int, default=0,
                        help="0 keeps every form")
    args = parser.parse_args()

    documents = load_funsd_images()
    if args.max_documents:
        documents = documents[: args.max_documents]
    sizes = {image.size for _, image in documents}
    print(f"loaded {len(documents)} real scans, sizes {sorted(sizes)}")

    manifest = []
    for index, (name, image) in enumerate(documents):
        folder = os.path.join(args.out, f"f{index:04d}")
        os.makedirs(folder, exist_ok=True)
        for layout_id, transform in enumerate(PASSES):
            path = os.path.join(folder, f"l{layout_id:02d}.png")
            if not os.path.exists(path):
                transform(image).save(path)
            manifest.append(
                {
                    "topic": "form",
                    "document": f"fun::{index:04d}",
                    "layout": layout_id,
                    "path": path,
                    "source": name,
                }
            )
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"real document domain: {len(manifest)} images, "
          f"{len({row['document'] for row in manifest})} forms, "
          f"{len({row['layout'] for row in manifest})} scan passes")


if __name__ == "__main__":
    main()
