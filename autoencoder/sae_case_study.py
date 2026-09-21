import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import torch

from backbone_survey import analyse, load_any_grid
from factor_eval import encode_all, load_models


GRIDS = (
    ("documents", "survey_grid"),
    ("shapes", "dataset/shape_grid"),
    ("real scans", "dataset/funsd_scan"),
    ("arxiv", "dataset/real_scan_wide"),
)


def main():
    parser = argparse.ArgumentParser(description="Released SAE, every domain.")
    parser.add_argument("--checkpoint-dir", default="model")
    parser.add_argument("--clip-weights",
                        default="clip_weights/open_clip_pytorch_model.bin")
    parser.add_argument("--sae-keys", default="logits_clip_img,sparse_codes_clip_img")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/sae_case_study.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sae_keys = [item.strip() for item in args.sae_keys.split(",") if item.strip()]
    models = load_models(args.checkpoint_dir, args.clip_weights, device)

    results = {}
    for domain, grid in GRIDS:
        paths, contents, layouts = load_any_grid(grid)
        features = encode_all(paths, sae_keys, models, device, args.batch_size)
        row = {}
        for key in ["clip"] + [f"sae:{name}" for name in sae_keys]:
            if key not in features:
                continue
            stats = analyse(features[key], contents, layouts)
            row[key] = {
                "ratio": stats["ratio"],
                "NN_R@1": stats["NN_R@1"],
                "spread": stats["per_dim_ratio_spread"],
                "oracle_diagonal": (
                    stats["ratio_after_diagonal"] / stats["ratio"] - 1.0
                ) * 100.0,
            }
            print(f"{domain:11s} {key:28s} ratio={stats['ratio']:5.2f} "
                  f"R@1={stats['NN_R@1']:.3f} spread={stats['per_dim_ratio_spread']:.3f} "
                  f"oracle={row[key]['oracle_diagonal']:+5.1f}%")
        results[domain] = row

    print("\nsparse / dense, same weights, same images")
    for domain, row in results.items():
        dense = row.get("sae:logits_clip_img")
        sparse = row.get("sae:sparse_codes_clip_img")
        if dense and sparse:
            print(f"  {domain:11s} ratio {dense['ratio']:5.2f} -> {sparse['ratio']:5.2f}   "
                  f"R@1 {dense['NN_R@1']:.3f} -> {sparse['NN_R@1']:.3f}   "
                  f"({sparse['NN_R@1'] / dense['NN_R@1'] - 1:+.0%})")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
