import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import json
import os
import re

from PIL import Image


EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_PATTERN = r"^(?P<doc>[A-Za-z0-9]+)_(?P<capture>\d+)"


def main():
    parser = argparse.ArgumentParser(description="Local paired-capture grid.")
    parser.add_argument("--root", default="dataset/midv_raw")
    parser.add_argument("--out", default="dataset/midv_full")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="regex with named groups doc and capture, matched "
                             "against the file stem")
    parser.add_argument("--min-captures", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--limit-documents", type=int, default=0,
                        help="0 keeps every document")
    args = parser.parse_args()

    pattern = re.compile(args.pattern)
    groups = collections.defaultdict(list)
    for base, _, names in os.walk(args.root):
        for name in sorted(names):
            if not name.lower().endswith(EXTENSIONS):
                continue
            match = pattern.match(os.path.splitext(name)[0])
            if match:
                groups[match.group("doc")].append(
                    (match.group("capture"), os.path.join(base, name))
                )
    kept = {doc: sorted(rows) for doc, rows in groups.items()
            if len(rows) >= args.min_captures}
    if args.limit_documents:
        kept = dict(sorted(kept.items())[: args.limit_documents])
    counts = [len(rows) for rows in kept.values()]
    print(f"{sum(len(v) for v in groups.values())} images, {len(groups)} documents, "
          f"{len(kept)} with at least {args.min_captures} captures")
    if not kept:
        raise SystemExit("no group survived: check --pattern with "
                         "inspect_capture_folder.py")
    print(f"captures per kept document: {min(counts)}-{max(counts)}")

    manifest = []
    for index, (doc, rows) in enumerate(sorted(kept.items())):
        folder = os.path.join(args.out, f"d{index:05d}")
        os.makedirs(folder, exist_ok=True)
        for layout, (capture, source) in enumerate(rows):
            path = os.path.join(folder, f"c{layout:02d}.png")
            if not os.path.exists(path):
                image = Image.open(source).convert("RGB")
                scale = args.max_side / max(image.size)
                if scale < 1:
                    image = image.resize(
                        (max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))), Image.BICUBIC)
                image.save(path)
            manifest.append({
                "topic": doc[:2],
                "document": f"{doc}",
                "layout": layout,
                "path": path,
                "source": os.path.relpath(source, args.root),
            })
        if (index + 1) % 100 == 0:
            print(f"  converted {index + 1}/{len(kept)}")
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"real-capture domain: {len(manifest)} images, {len(kept)} documents, "
          f"{len({r['layout'] for r in manifest})} captures")
    print(f"manifest: {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
