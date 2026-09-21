import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import os
import re


EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")

PATTERNS = (
    ("midv-style <doc>_<type>", r"^(?P<doc>[A-Za-z0-9]+)_(?P<capture>\d+)$"),
    ("<doc>_<capture>_suffix", r"^(?P<doc>[A-Za-z0-9]+)_(?P<capture>\d+)[_-].*$"),
    ("<doc>-<capture>", r"^(?P<doc>[A-Za-z0-9]+)-(?P<capture>\d+)$"),
    ("<doc><capture>", r"^(?P<doc>[A-Za-z]+)(?P<capture>\d+)$"),
    ("parent folder is the document", r"^(?P<capture>.*)$"),
)


def main():
    parser = argparse.ArgumentParser(description="Inspect a downloaded folder.")
    parser.add_argument("--root", default="dataset/midv_raw")
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args()

    files = []
    for base, _, names in os.walk(args.root):
        for name in sorted(names):
            if name.lower().endswith(EXTENSIONS):
                files.append(os.path.join(base, name))
    print(f"{len(files)} image files under {args.root}")
    if not files:
        return
    print("sample names:")
    for path in files[: args.max_samples]:
        print("   ", os.path.relpath(path, args.root))
    rel = [os.path.relpath(path, args.root) for path in files]
    depths = collections.Counter(len(os.path.dirname(name).split(os.sep)) for name in rel)
    print("directory depth histogram:", dict(depths))
    sizes = collections.Counter(os.path.splitext(name)[1].lower() for name in rel)
    print("extensions:", dict(sizes))

    print("\ncandidate groupings")
    for label, pattern in PATTERNS:
        groups = collections.defaultdict(list)
        compiled = re.compile(pattern)
        for path, name in zip(files, rel):
            stem = os.path.splitext(name)[0]
            if pattern.startswith("^(?P<capture>"):
                parent = os.path.basename(os.path.dirname(name))
                match = compiled.match(stem)
                if match:
                    groups[parent].append(match.group("capture"))
                continue
            match = compiled.match(stem)
            if match:
                groups[match.group("doc")].append(match.group("capture"))
        if not groups:
            continue
        counts = [len(v) for v in groups.values()]
        usable = sum(1 for c in counts if c >= 3)
        print(f"  {label:34s} documents {len(groups):6d}  "
              f"captures/doc {min(counts)}-{max(counts)}  docs with >=3 {usable}")


if __name__ == "__main__":
    main()
