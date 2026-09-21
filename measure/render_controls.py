import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import torch

from dataset_generator import run_dataset_generation
from experiment_utils import metrics
from seed_variance import cached_features


SETTINGS = (
    ("short text, sized to text", dict(min_chars=50, max_chars=400, canvas_height=0,
                                       fixed_length=0)),
    ("medium text, sized to text", dict(min_chars=300, max_chars=600, canvas_height=0,
                                        fixed_length=0)),
    ("fixed length, sized to text", dict(min_chars=300, max_chars=600, canvas_height=0,
                                         fixed_length=400)),
    ("fixed canvas height", dict(min_chars=300, max_chars=600, canvas_height=1024,
                                 fixed_length=400)),
    ("fixed canvas, short text", dict(min_chars=50, max_chars=200, canvas_height=1024,
                                      fixed_length=0)),
)


def main():
    parser = argparse.ArgumentParser(description="Rendering controls.")
    parser.add_argument("--corpus", default="train-00000-of-00002.parquet")
    parser.add_argument("--contents", type=int, default=20)
    parser.add_argument("--layouts", type=int, default=4)
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--out-root", default="dataset/render_controls")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/render_controls.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    results = {}

    for name, config in SETTINGS:
        slug = name.replace(" ", "_")
        grid = os.path.join(args.out_root, slug)
        if not os.path.exists(os.path.join(grid, "layout_manifest.json")):
            os.makedirs(grid, exist_ok=True)
            print(f"\n=== rendering {name}")
            run_dataset_generation(
                args.corpus, base_output_dir=grid, total_goal=args.contents,
                variations=args.layouts, seed=42, layout_grid=args.layouts,
                layouts_file=os.path.join(grid, "layouts.json"), **config,
            )
        row = {}
        for encoder in encoders:
            features, contents, layouts = cached_features(
                grid, f"render_{slug}", encoder, device, "outputs/cache"
            )
            stats = metrics(features, contents, layouts)
            row[encoder] = stats
            print(f"{name:30s} {encoder:12s} ratio={stats['ratio']:5.2f} "
                  f"R@1={stats['NN_R@1']:.3f}")
        results[name] = row

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
