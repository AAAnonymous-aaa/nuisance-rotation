import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from factor_eval import cosine_distance
from experiment_utils import fit_basis, load, project


def recall_from_distance(distance, contents, layouts, gallery_layout=0):
    gallery = np.where(layouts == gallery_layout)[0]
    if len(gallery) == 0:
        return float("nan")
    hits = total = 0
    for index in range(len(contents)):
        if layouts[index] == gallery_layout:
            continue
        row = distance[index, gallery]
        if contents[gallery[int(np.argmin(row))]] == contents[index]:
            hits += 1
        total += 1
    return hits / total if total else float("nan")


def bootstrap(distance, contents, layouts, rounds, rng):
    unique = np.unique(contents)
    index_of = {c: np.where(contents == c)[0] for c in unique}
    values = []
    for _ in range(rounds):
        picked = rng.choice(unique, size=len(unique), replace=True)


        index = np.concatenate([index_of[c] for c in sorted(set(picked.tolist()))])
        sub_contents = contents[index]
        sub_layouts = layouts[index]
        sub = distance[np.ix_(index, index)]
        values.append(recall_from_distance(sub, sub_contents, sub_layouts))
    return np.asarray(values, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap for retrieval.")
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--domains", default="documents,arxiv")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="outputs/retrieval_ci.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = {}
    for domain in domains:
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            basis = fit_basis(features, contents, layouts, rank=args.rank)
            rotated = project(features, basis, args.rank)

            def distance_of(matrix):
                base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
                return cosine_distance(base).numpy()

            raw_distance = distance_of(features)
            rot_distance = distance_of(rotated)
            raw = bootstrap(raw_distance, contents, layouts, args.rounds,
                            np.random.default_rng(args.seed))
            rot = bootstrap(rot_distance, contents, layouts, args.rounds,
                            np.random.default_rng(args.seed))
            delta = rot - raw
            row = {
                "raw_mean": float(np.mean(raw)),
                "rotated_mean": float(np.mean(rot)),
                "delta_mean": float(np.mean(delta)),
                "delta_ci95": [float(np.percentile(delta, 2.5)),
                               float(np.percentile(delta, 97.5))],
                "delta_positive_fraction": float(np.mean(delta > 0)),
            }
            results[f"{domain}/{encoder}"] = row
            print(f"{domain:12s} {encoder:14s} raw {row['raw_mean']:.3f}  "
                  f"rotated {row['rotated_mean']:.3f}  "
                  f"delta {row['delta_mean']:+.3f} "
                  f"[{row['delta_ci95'][0]:+.3f}, {row['delta_ci95'][1]:+.3f}]")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
