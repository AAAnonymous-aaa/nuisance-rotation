import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from experiment_utils import DOMAINS, load
from factor_eval import (
    cosine_distance,
    nearest_neighbour_recall,
    pair_means,
    variance_decomposition,
)


SURVEYS = {
    "documents": "backbone_survey.json",
    "shapes": "shape_survey.json",
    "real scans": "funsd_scan_survey.json",
    "large scans": "docvqa_scan_survey.json",
    "arxiv": "real_scan_wide_survey.json",
    "identity documents": "midv_full_survey.json",
}


def ratio_of(weights, features, contents, layouts):

    weighted = features * weights
    weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
    similarity = weighted @ weighted.t()
    distance = 1.0 - similarity
    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    same_content = torch.tensor(contents[:, None] == contents[None, :])
    same_layout = torch.tensor(layouts[:, None] == layouts[None, :])
    layout_pairs = distance[same_content & ~same_layout].mean()
    content_pairs = distance[~same_content & same_layout].mean()
    return content_pairs / layout_pairs.clamp_min(1e-12)


def main():
    parser = argparse.ArgumentParser(description="Numerical ceiling of the family.")
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--report-json", default="outputs/diagonal_ceiling.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in SURVEYS:
            continue
        survey = json.load(open(os.path.join("outputs", SURVEYS[domain]), encoding="utf-8"))
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            if len(features) > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features, contents, layouts = features[mask], contents[mask], layouts[mask]
            raw = ratio_of(torch.ones(features.shape[1]), features, contents, layouts)

            layout_var, content_var = variance_decomposition(features, contents, layouts)
            closed = (content_var / (layout_var + 1e-12))
            closed = (closed / closed.mean()).clamp(0.1, 10.0).sqrt()
            closed_ratio = ratio_of(closed, features, contents, layouts)

            log_weights = torch.zeros(features.shape[1], requires_grad=True)
            optimizer = torch.optim.Adam([log_weights], lr=args.lr)
            for step in range(args.steps):
                optimizer.zero_grad()
                value = ratio_of(log_weights.exp(), features, contents, layouts)
                loss = -value
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                optimised = ratio_of(log_weights.exp(), features, contents, layouts)
                optimised_weights = log_weights.exp()

            def retrieval(weights):
                weighted = features * weights
                weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
                distance = cosine_distance(weighted)
                return float(nearest_neighbour_recall(distance, contents, layouts))

            row = {
                "raw": float(raw),
                "closed_form_oracle": float(closed_ratio),
                "numerical_optimum": float(optimised),
                "retrieval_raw": retrieval(torch.ones(features.shape[1])),
                "retrieval_closed_form": retrieval(closed),
                "retrieval_numerical": retrieval(optimised_weights),
                "weight_dynamic_range": float(
                    (optimised_weights.max() / optimised_weights.min()).item()),
                "closed_form_gain_percent": float((closed_ratio / raw - 1.0) * 100.0),
                "numerical_gain_percent": float((optimised / raw - 1.0) * 100.0),
                "headroom_above_closed_form_percent": float(
                    (optimised / closed_ratio - 1.0) * 100.0),
                "published_oracle_percent": float(
                    (survey[encoder]["ratio_after_diagonal"] /
                     survey[encoder]["ratio"] - 1.0) * 100.0),
            }
            results[f"{domain}/{encoder}"] = row
            print(f"{domain:18s} {encoder:14s} raw {row['raw']:5.2f}  "
                  f"closed {row['closed_form_oracle']:5.2f} ({row['closed_form_gain_percent']:+5.1f}%)  "
                  f"optimised {row['numerical_optimum']:5.2f} "
                  f"({row['numerical_gain_percent']:+5.1f}%)  "
                  f"room {row['headroom_above_closed_form_percent']:+6.1f}%  "
                  f"R@1 raw/closed/opt "
                  f"{row['retrieval_raw']:.3f}/{row['retrieval_closed_form']:.3f}/"
                  f"{row['retrieval_numerical']:.3f}  weight range "
                  f"{row['weight_dynamic_range']:.0f}x")

    closed = np.array([r["closed_form_gain_percent"] for r in results.values()])
    optimised = np.array([r["numerical_gain_percent"] for r in results.values()])
    room = np.array([r["headroom_above_closed_form_percent"] for r in results.values()])
    summary = {
        "n": len(results),
        "median_room_above_closed_form_percent": float(np.median(room)),
        "max_room_above_closed_form_percent": float(room.max()),
        "corr_closed_optimised": float(np.corrcoef(closed, optimised)[0, 1]),
    }
    print("\nsummary:", json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": results, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
