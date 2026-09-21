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
    distance = 1.0 - weighted @ weighted.t()
    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    same_content = torch.tensor(contents[:, None] == contents[None, :])
    same_layout = torch.tensor(layouts[:, None] == layouts[None, :])
    layout_pairs = distance[same_content & ~same_layout].mean()
    content_pairs = distance[~same_content & same_layout].mean()
    return content_pairs / layout_pairs.clamp_min(1e-12)


def recall(weights, features, contents, layouts):
    weighted = features * weights
    weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
    distance = cosine_distance(weighted)
    return float(nearest_neighbour_recall(distance, contents, layouts))


def optimise(features, contents, layouts, clip, steps, lr, seed=0):

    torch.manual_seed(seed)
    bound = float(np.log(clip))
    log_weights = torch.zeros(features.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([log_weights], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        value = ratio_of(log_weights.exp(), features, contents, layouts)
        (-value).backward()
        optimizer.step()
        with torch.no_grad():
            log_weights.clamp_(-bound, bound)
    with torch.no_grad():
        return log_weights.exp().detach()


def main():
    parser = argparse.ArgumentParser(description="Bounded-dynamic-range frontier.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--clips", default="1,2,4,10,40,200")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--report-json", default="outputs/diagonal_frontier.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    clips = [float(c) for c in args.clips.split(",") if c.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in SURVEYS:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            if len(features) > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features, contents, layouts = features[mask], contents[mask], layouts[mask]
            ones = torch.ones(features.shape[1])
            raw_ratio = float(ratio_of(ones, features, contents, layouts))
            raw_recall = recall(ones, features, contents, layouts)

            layout_var, content_var = variance_decomposition(features, contents, layouts)
            closed = (content_var / (layout_var + 1e-12))
            closed = (closed / closed.mean()).clamp(0.1, 10.0).sqrt()
            closed_ratio = float(ratio_of(closed, features, contents, layouts))
            closed_recall = recall(closed, features, contents, layouts)

            row = {"raw_ratio": raw_ratio, "raw_recall": raw_recall,
                   "closed_form_ratio": closed_ratio,
                   "closed_form_recall": closed_recall,
                   "frontier": {}}
            for clip in clips:
                weights = optimise(features, contents, layouts, clip, args.steps, args.lr)
                row["frontier"][f"clip={clip:g}"] = {
                    "ratio": float(ratio_of(weights, features, contents, layouts)),
                    "recall": recall(weights, features, contents, layouts),
                    "gain_percent": float(
                        (ratio_of(weights, features, contents, layouts) / raw_ratio - 1) * 100),
                }
            results[f"{domain}/{encoder}"] = row
            frontier = " ".join(
                f"c={key.split('=')[1]}:{value['gain_percent']:+.0f}%/{value['recall']:.2f}"
                for key, value in row["frontier"].items())
            print(f"{domain:18s} {encoder:12s} raw {raw_ratio:5.2f}/{raw_recall:.3f}  "
                  f"closed {closed_ratio:5.2f}/{closed_recall:.3f}  | {frontier}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
