import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from experiment_utils import DOMAINS, load, metrics, project
from factor_eval import layout_differences, split_by_content


RANKS = (1, 2, 4, 8, 16, 32)


def main():
    parser = argparse.ArgumentParser(description="Automatic rank selection.")
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--domains",
                        default="documents,shapes,real scans,large scans,arxiv,"
                                "identity documents")
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--report-json", default="outputs/rank_selection.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    wanted = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in wanted:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            left, right = split_by_content(contents)
            calibration = layout_differences(
                features[left], contents[left], layouts[left], n_pairs=4000
            )
            torch.manual_seed(0)
            basis = torch.pca_lowrank(calibration, q=max(RANKS), center=False)[2]
            spectrum = torch.linalg.svdvals(calibration) ** 2
            cumulative = np.cumsum(spectrum.numpy()) / spectrum.sum().item()
            energy_rank = int(np.searchsorted(cumulative, args.energy) + 1)

            selection_scores = {}
            for rank in RANKS:
                row = metrics(project(features[left], basis, rank),
                              contents[left], layouts[left])
                selection_scores[rank] = row["NN_R@1"]
            heldout_rank = max(selection_scores, key=selection_scores.get)

            def score(rank):
                return metrics(project(features[right], basis, rank),
                               contents[right], layouts[right])["NN_R@1"]

            results[f"{domain}/{encoder}"] = {
                "energy_rank": energy_rank,
                "heldout_rank": int(heldout_rank),
                "fixed_r4": score(4),
                "energy_rank_score": score(energy_rank),
                "heldout_rank_score": score(int(heldout_rank)),
                "selection_scores": {str(k): v for k, v in selection_scores.items()},
            }
            row = results[f"{domain}/{encoder}"]
            print(f"{domain:14s} {encoder:14s} energy r={energy_rank:2d} "
                  f"chosen r={heldout_rank:2d} | held-out R@1: "
                  f"fixed4 {row['fixed_r4']:.3f}  energy {row['energy_rank_score']:.3f}  "
                  f"chosen {row['heldout_rank_score']:.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
