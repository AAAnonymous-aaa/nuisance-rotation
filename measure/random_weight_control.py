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


def main():
    parser = argparse.ArgumentParser(description="Random reweighting control.")
    parser.add_argument("--encoders", default="clip_openai")
    parser.add_argument("--clips", default="2,4,10")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--report-json", default="outputs/random_weight_control.json")
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
            rng = np.random.default_rng(0)
            row = {"raw_ratio": raw_ratio, "raw_recall": raw_recall, "clips": {}}
            for clip in clips:
                gains, recalls = [], []
                for _ in range(args.draws):
                    logs = rng.uniform(-np.log(clip), np.log(clip), size=features.shape[1])
                    weights = torch.tensor(np.exp(logs), dtype=torch.float32)
                    gains.append((float(ratio_of(weights, features, contents, layouts))
                                  / raw_ratio - 1.0) * 100.0)
                    recalls.append(recall(weights, features, contents, layouts))
                row["clips"][f"clip={clip:g}"] = {
                    "gain_mean": float(np.mean(gains)),
                    "gain_sd": float(np.std(gains)),
                    "gain_min": float(np.min(gains)),
                    "gain_max": float(np.max(gains)),
                    "recall_mean": float(np.mean(recalls)),
                    "recall_min": float(np.min(recalls)),
                }
            results[f"{domain}/{encoder}"] = row
            text = "  ".join(
                f"c={k.split('=')[1]}: {v['gain_mean']:+.0f}+/-{v['gain_sd']:.0f}% "
                f"(R@1 {v['recall_mean']:.3f}, min {v['recall_min']:.3f})"
                for k, v in row["clips"].items())
            print(f"{domain:18s} raw {raw_ratio:5.2f}/{raw_recall:.3f}  {text}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
