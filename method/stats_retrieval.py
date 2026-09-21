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


def distance_of(matrix):
    base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
    return cosine_distance(base).numpy()


def bootstrap(distance, contents, layouts, rounds, rng):
    unique = np.unique(contents)
    index_of = {c: np.where(contents == c)[0] for c in unique}
    layouts = np.asarray(layouts)
    values = np.empty(rounds)
    for step in range(rounds):
        picked = np.unique(rng.choice(unique, size=len(unique), replace=True))
        index = np.concatenate([index_of[c] for c in picked])
        gallery = index[layouts[index] == 0]
        queries = index[layouts[index] != 0]
        if len(gallery) == 0 or len(queries) == 0:
            values[step] = np.nan
            continue
        sub = distance[np.ix_(queries, gallery)]
        nearest = sub.argmin(axis=1)
        hits = contents[gallery[nearest]] == contents[queries]
        values[step] = hits.mean()
    return values


def main():
    parser = argparse.ArgumentParser(description="Retrieval effect sizes.")
    parser.add_argument("--encoders", default="clip_openai,siglip,clip_datacomp")
    parser.add_argument("--domains", default="documents,shapes,arxiv")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--fit-seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--report-json", default="outputs/retrieval_stats.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = {}
    for domain in domains:
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            raw_distance = distance_of(features)
            rng = np.random.default_rng(0)
            raw = bootstrap(raw_distance, contents, layouts, args.rounds, rng)
            deltas = []
            for seed in range(args.fit_seeds):
                basis = fit_basis(features, contents, layouts, rank=args.rank,
                                  mask=None, seed=seed)
                rotated = project(features, basis, args.rank)
                rng = np.random.default_rng(0)
                values = bootstrap(distance_of(rotated), contents, layouts,
                                   args.rounds, rng)
                deltas.append(values - raw)
            delta = np.mean(np.stack(deltas), axis=0)
            lo, hi = np.percentile(delta, [2.5, 97.5])
            effect = float(delta.mean() / (delta.std() + 1e-12))
            row = {
                "raw": float(np.nanmean(raw)),
                "delta_mean": float(delta.mean()),
                "delta_ci95": [float(lo), float(hi)],
                "p_positive": float(np.mean(delta > 0)),
                "effect_size": effect,
                "significant": bool(lo > 0),
                "fit_seeds": args.fit_seeds,
                "rounds": args.rounds,
            }
            results[f"{domain}/{encoder}"] = row
            mark = "*" if row["significant"] else " "
            print(f"{domain:14s} {encoder:14s} raw {row['raw']:.3f}  "
                  f"delta {row['delta_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark} "
                  f"P(+)={row['p_positive']:.3f}  d={effect:+.2f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
