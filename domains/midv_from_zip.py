import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import json
import os
import re
import zipfile

import numpy as np
from PIL import Image


ENTRY = re.compile(r"^(?P<type>[^/]+)/images/(?P<group>[^/]+)/"
                   r"(?P<doc>[A-Za-z]+[0-9]+)_(?P<capture>[0-9]+)\.tif$")


def main():
    parser = argparse.ArgumentParser(description="MIDV-500 from archive.zip.")
    parser.add_argument("--zip", default="archive.zip")
    parser.add_argument("--out", default="dataset/midv_full")
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--min-captures", type=int, default=3)
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.zip)
    groups = collections.defaultdict(list)
    overviews = 0
    for info in archive.infolist():
        if not info.filename.lower().endswith(".tif"):
            continue
        match = ENTRY.match(info.filename)
        if not match:
            overviews += 1
            continue
        groups[(match.group("type"), match.group("doc"))].append(
            (int(match.group("capture")), match.group("group"), info.filename)
        )
    kept = {key: sorted(rows) for key, rows in groups.items()
            if len(rows) >= args.min_captures}
    counts = [len(rows) for rows in kept.values()]
    print(f"{sum(len(v) for v in groups.values())} capture images over "
          f"{len(groups)} documents ({overviews} overview images skipped)")
    print(f"{len(kept)} documents with >= {args.min_captures} captures "
          f"({min(counts)}-{max(counts)} each); "
          f"{len({k[0] for k in kept})} document types")

    manifest = []
    shapes = collections.Counter()
    modes = collections.Counter()
    for index, (key, rows) in enumerate(sorted(kept.items())):
        doc_type, doc = key
        folder = os.path.join(args.out, f"d{index:04d}")
        os.makedirs(folder, exist_ok=True)
        for layout, (capture, group, source) in enumerate(rows):
            path = os.path.join(folder, f"c{layout:02d}.jpg")
            if not os.path.exists(path):
                with archive.open(source) as handle:
                    image = Image.open(handle).convert("RGB")
                scale = args.max_side / max(image.size)
                if scale < 1:
                    image = image.resize(
                        (max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))), Image.BICUBIC)
                image.save(path, quality=88)
            with Image.open(path) as check:
                shapes[layout] = check.size
                modes[layout] = check.mode
            manifest.append({
                "topic": doc_type,
                "document": f"{doc_type}::{doc}",
                "layout": layout,
                "path": path,
                "source": source,
                "capture_id": capture,
            })
        if (index + 1) % 100 == 0:
            print(f"  converted {index + 1}/{len(kept)} documents")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("image size and mode per capture index (first five documents):")
    for layout in sorted(shapes):
        print(f"  capture {layout}: {shapes[layout]} {modes[layout]}")
    print(f"real-capture domain: {len(manifest)} images, {len(kept)} documents, "
          f"{len({r['layout'] for r in manifest})} captures")
    print(f"manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
