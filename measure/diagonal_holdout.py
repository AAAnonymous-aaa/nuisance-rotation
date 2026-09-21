import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from diagonal_frontier import SURVEYS, ratio_of, recall
from experiment_utils import DOMAINS, load
from factor_eval import split_by_content, variance_decomposition


def as_mask(selection, n):
    selection = np.asarray(selection)
    if selection.dtype == bool:
        return selection
    mask = np.zeros(n, dtype=bool)
    mask[selection] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description="Held-out optimised diagonal.")
    parser.add_argument("--encoders", default="clip_openai")
    parser.add_argument("--clips", default="2,4,10")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--report-json", default="outputs/diagonal_holdout.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    clips = [float(c) for c in args.clips.split(",") if c.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in SURVEYS:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            n = len(features)
            left = as_mask(split_by_content(contents)[0], n)
            right = ~left
            row = {
                "raw_ratio": float(ratio_of(torch.ones(features.shape[1]),
                                            features[right], contents[right], layouts[right])),
                "raw_recall": recall(torch.ones(features.shape[1]), features[right],
                                     contents[right], layouts[right]),
                "clips": {},
            }

            layout_var, content_var = variance_decomposition(
                features[left], contents[left], layouts[left])
            plug = (content_var / (layout_var + 1e-12))
            plug = (plug / plug.mean()).clamp(0.1, 10.0).sqrt()
            plug_ratio = float(ratio_of(plug, features[right], contents[right], layouts[right]))
            row["plugin_ratio"] = plug_ratio
            row["plugin_gain_percent"] = (plug_ratio / row["raw_ratio"] - 1.0) * 100.0
            row["plugin_recall"] = recall(plug, features[right], contents[right], layouts[right])

            for clip in clips:
                bound = float(np.log(clip))
                log_weights = torch.zeros(features.shape[1], requires_grad=True)
                optimizer = torch.optim.Adam([log_weights], lr=args.lr)
                for _ in range(args.steps):
                    optimizer.zero_grad()
                    value = ratio_of(log_weights.exp(), features[left], contents[left],
                                     layouts[left])
                    (-value).backward()
                    optimizer.step()
                    with torch.no_grad():
                        log_weights.clamp_(-bound, bound)
                with torch.no_grad():
                    weights = log_weights.exp()
                ratio = float(ratio_of(weights, features[right], contents[right], layouts[right]))
                row["clips"][f"clip={clip:g}"] = {
                    "ratio": ratio,
                    "gain_percent": (ratio / row["raw_ratio"] - 1.0) * 100.0,
                    "recall": recall(weights, features[right], contents[right], layouts[right]),
                }
            results[f"{domain}/{encoder}"] = row
            text = "  ".join(
                f"c={k.split('=')[1]}: {v['gain_percent']:+.0f}%/R@1 {v['recall']:.3f}"
                for k, v in row["clips"].items())
            print(f"{domain:18s} raw {row['raw_ratio']:5.2f}/{row['raw_recall']:.3f}  "
                  f"plugin {row['plugin_gain_percent']:+5.1f}%/R@1 {row['plugin_recall']:.3f}  |"
                  f" optimised held out: {text}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
