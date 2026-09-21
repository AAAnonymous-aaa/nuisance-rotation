import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import torch

from backbone_survey import analyse
from dataset_generator import get_few_valid_fonts, run_dataset_generation
from seed_variance import cached_features


LEVELS = (0.15, 0.35, 0.6, 1.0)


def layouts_for(level, font_path):

    sizes = [max(10, round(22 - 6 * level)), round(22 + 6 * level)]
    widths = [round(750 - 150 * level), round(750 + 150 * level)]
    margin = round(40 - 15 * level)
    spacing = round(1.35 - 0.2 * level, 2)
    layouts = []
    for size in sizes:
        for width in widths:
            layouts.append(
                {
                    "font_path": font_path,
                    "font_size": size,
                    "image_width": width,
                    "margin": margin,
                    "line_spacing": spacing,
                    "align": "left",
                }
            )
    return layouts


def main():
    parser = argparse.ArgumentParser(description="Layout strength sweep.")
    parser.add_argument("--corpus", default="train-00000-of-00002.parquet")
    parser.add_argument("--contents", type=int, default=30)
    parser.add_argument("--fixed-length", type=int, default=400)
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--out-root", default="dataset/strength")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/strength_sweep.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoders = [item.strip() for item in args.encoders.split(",") if item.strip()]
    font = get_few_valid_fonts(required_count=20)[0]

    grids = []
    for level in LEVELS:
        grid = os.path.join(args.out_root, f"t{level:.2f}")
        layouts_file = os.path.join(grid, "layouts.json")
        if not os.path.exists(os.path.join(grid, "layout_manifest.json")):
            os.makedirs(grid, exist_ok=True)
            with open(layouts_file, "w", encoding="utf-8") as handle:
                json.dump(layouts_for(level, font), handle, ensure_ascii=False, indent=2)
            print(f"\n=== strength {level}: rendering {grid}")
            run_dataset_generation(
                args.corpus,
                base_output_dir=grid,
                total_goal=args.contents,
                variations=4,
                seed=42,
                layout_grid=4,
                layouts_file=layouts_file,
                min_chars=300,
                max_chars=600,
                canvas_height=0,
                fixed_length=args.fixed_length,
            )
        grids.append((level, grid))

    results = {}
    for level, grid in grids:
        name = f"t{level:.2f}"
        row = {}
        for encoder in encoders:
            features, contents, layouts = cached_features(
                grid, name, encoder, device, args.cache_dir
            )
            stats = analyse(features, contents, layouts)
            row[encoder] = {
                "ratio": stats["ratio"],
                "NN_R@1": stats["NN_R@1"],
                "spread": stats["per_dim_ratio_spread"],
                "oracle_diagonal": (
                    stats["ratio_after_diagonal"] / stats["ratio"] - 1.0
                ) * 100.0,
                "rotation_r4": (
                    stats["ratio_after_rotation"][4] / stats["ratio"] - 1.0
                ) * 100.0,
            }
            print(f"{name:6s} {encoder:12s} ratio={stats['ratio']:5.2f} "
                  f"R@1={stats['NN_R@1']:.3f} spread={stats['per_dim_ratio_spread']:.3f} "
                  f"oracle={row[encoder]['oracle_diagonal']:+5.1f}% "
                  f"rot={row[encoder]['rotation_r4']:+6.1f}%")
        results[name] = row

    print("\nspread -> oracle headroom")
    for level, _ in grids:
        for encoder, row in results[f"t{level:.2f}"].items():
            print(f"  {level:.2f} {encoder:12s} spread={row['spread']:.3f} "
                  f"oracle={row['oracle_diagonal']:+5.1f}%")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
