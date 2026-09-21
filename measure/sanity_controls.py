import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from factor_eval import cosine_distance, pair_means, variance_decomposition
from experiment_utils import half_of, load, metrics


def main():
    parser = argparse.ArgumentParser(description="Layout count and held-out oracle.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--domains", default="documents,shapes,arxiv")
    parser.add_argument("--report-json", default="outputs/sanity_controls.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = {}

    for domain in domains:
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            row = {}
            available = sorted(set(layouts.tolist()))
            for count in (2, 3, 4, len(available)):
                if count > len(available):
                    continue
                keep = np.isin(layouts, available[:count])
                row[f"layouts={count}"] = metrics(
                    features[keep], contents[keep], layouts[keep]
                )

            mask = half_of(contents)
            layout_var, content_var = variance_decomposition(
                features[mask], contents[mask], layouts[mask]
            )
            weights = content_var / (layout_var + 1e-12)
            weights = (weights / (weights.mean() + 1e-12)).clamp(0.1, 10.0).sqrt()
            base = features[~mask]
            base = base / (base.norm(dim=-1, keepdim=True) + 1e-9)
            weighted = base * weights
            weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
            distance = cosine_distance(weighted)
            s_layout, s_content, _ = pair_means(
                distance, contents[~mask], layouts[~mask]
            )
            ratio_after = s_content / s_layout
            ratio_before = metrics(base, contents[~mask], layouts[~mask])["ratio"]
            row["held_out_oracle_diagonal"] = {
                "ratio_before": ratio_before,
                "ratio_after": float(ratio_after),
                "gain_percent": float((ratio_after / ratio_before - 1.0) * 100.0),
            }
            results[f"{domain}/{encoder}"] = row
            layout_line = " ".join(
                f"{k} r={v['ratio']:.2f}/R@1={v['NN_R@1']:.2f}"
                for k, v in row.items() if k.startswith("layouts")
            )
            print(f"{domain:12s} {encoder:12s} {layout_line} | "
                  f"held-out oracle {row['held_out_oracle_diagonal']['gain_percent']:+.1f}%")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
