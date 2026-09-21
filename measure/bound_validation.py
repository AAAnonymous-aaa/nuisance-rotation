import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from experiment_utils import metrics


def synth(contents, layouts, dim, spread, seed=0):


    rng = np.random.default_rng(seed)
    shared = np.exp(0.5 * rng.standard_normal(dim)).astype(np.float32)
    alpha = shared * np.exp(spread * rng.standard_normal(dim)).astype(np.float32)
    beta = shared * np.exp(spread * rng.standard_normal(dim)).astype(np.float32)
    alpha /= alpha.mean()
    beta /= beta.mean()

    content_vector = rng.standard_normal((len(set(contents.tolist())), dim)).astype(np.float32)
    content_vector *= alpha[None, :]
    layout_vector = rng.standard_normal((len(set(layouts.tolist())), dim)).astype(np.float32)
    layout_vector *= beta[None, :]

    content_ids = {c: i for i, c in enumerate(sorted(set(contents.tolist())))}
    layout_ids = {l: i for i, l in enumerate(sorted(set(layouts.tolist())))}
    rows = []
    for c, l in zip(contents, layouts):
        rows.append(content_vector[content_ids[c]] + layout_vector[layout_ids[l]]
                    + 0.3 * rng.standard_normal(dim).astype(np.float32))
    return torch.tensor(np.stack(rows))


def main():
    parser = argparse.ArgumentParser(description="Synthetic validation of the bound.")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--contents", type=int, default=84)
    parser.add_argument("--layouts", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--report-json", default="outputs/bound_validation.json")
    args = parser.parse_args()

    contents = np.array([f"c{i:03d}" for i in range(args.contents)
                         for _ in range(args.layouts)])
    layouts = np.array([l for _ in range(args.contents)
                        for l in range(args.layouts)])

    results = {"sigma_sweep": {}, "sample_floor": {}, "layout_floor": {}}

    print("sampling floor: true spread is zero at sigma=0")
    for count in (84, 168, 336, 672):
        contents = np.array([f"c{i:04d}" for i in range(count)
                             for _ in range(args.layouts)])
        layouts = np.array([l for _ in range(count)
                            for l in range(args.layouts)])
        spreads, headrooms = [], []
        for seed in range(args.seeds):
            features = synth(contents, layouts, args.dim, 0.0, seed=seed)
            base = features / (features.norm(dim=-1, keepdim=True) + 1e-9)
            from factor_eval import (cosine_distance, pair_means,
                                     variance_decomposition)
            distance = cosine_distance(base)
            s_layout, s_content, _ = pair_means(distance, contents, layouts)
            ratio = s_content / s_layout
            layout_var, content_var = variance_decomposition(features, contents,
                                                             layouts)
            per_dim = (content_var / (layout_var + 1e-12)).numpy()
            spreads.append(float(np.std(per_dim) / (np.mean(per_dim) + 1e-12)))
            weights = (content_var / (layout_var + 1e-12))
            weights = (weights / (weights.mean() + 1e-12)).clamp(0.1, 10.0).sqrt()
            weighted = base * weights
            weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
            d2 = cosine_distance(weighted)
            sl2, sc2, _ = pair_means(d2, contents, layouts)
            headrooms.append((sc2 / sl2) / ratio - 1.0)
        results["sample_floor"][f"contents={count}"] = {
            "spread": float(np.mean(spreads)),
            "spread_sd": float(np.std(spreads)),
            "headroom": float(np.mean(headrooms)),
        }
        print(f"  contents={count:5d}  measured spread={np.mean(spreads):.3f} "
              f"(sd {np.std(spreads):.3f})  headroom={np.mean(headrooms) * 100:+5.1f}%")

    contents = np.array([f"c{i:03d}" for i in range(args.contents)
                         for _ in range(args.layouts)])
    layouts = np.array([l for _ in range(args.contents)
                        for l in range(args.layouts)])
    print("\nlayout floor: true spread is zero, but the nuisance variance is "
          "estimated from L layouts")
    for layouts_count in (4, 6, 12, 24, 48):
        contents = np.array([f"c{i:003d}" for i in range(84)
                             for _ in range(layouts_count)])
        layouts = np.array([l for _ in range(84)
                            for l in range(layouts_count)])
        spreads = []
        for seed in range(args.seeds):
            features = synth(contents, layouts, args.dim, 0.0, seed=seed)
            from factor_eval import variance_decomposition
            layout_var, content_var = variance_decomposition(features, contents,
                                                             layouts)
            per_dim = (content_var / (layout_var + 1e-12)).numpy()
            spreads.append(float(np.std(per_dim) / (np.mean(per_dim) + 1e-12)))
        results["layout_floor"][f"layouts={layouts_count}"] = {
            "spread": float(np.mean(spreads)), "spread_sd": float(np.std(spreads))}
        print(f"  layouts={layouts_count:3d}  measured spread={np.mean(spreads):.3f} "
              f"(sd {np.std(spreads):.3f})")

    sweep_contents = 84
    sweep_realisations = 48
    contents = np.array([f"c{i:003d}" for i in range(sweep_contents)
                         for _ in range(sweep_realisations)])
    layouts = np.array([l for _ in range(sweep_contents)
                        for l in range(sweep_realisations)])
    print(f"\nspread sweep at {sweep_contents} contents x {sweep_realisations} "
          "realisations, where the estimator floor is small enough to expose the "
          "true relationship")
    for spread in (0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6):
        spreads, headrooms = [], []
        for seed in range(args.seeds):
            features = synth(contents, layouts, args.dim, spread, seed=seed)
            base = features / (features.norm(dim=-1, keepdim=True) + 1e-9)
            from factor_eval import (cosine_distance, pair_means,
                                     variance_decomposition)
            distance = cosine_distance(base)
            s_layout, s_content, _ = pair_means(distance, contents, layouts)
            ratio = s_content / s_layout
            layout_var, content_var = variance_decomposition(features, contents, layouts)
            per_dim = (content_var / (layout_var + 1e-12)).numpy()
            spreads.append(float(np.std(per_dim) / (np.mean(per_dim) + 1e-12)))
            weights = (content_var / (layout_var + 1e-12))
            weights = weights / (weights.mean() + 1e-12)
            weights = weights.clamp(0.1, 10.0).sqrt()
            weighted = base * weights
            weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
            d2 = cosine_distance(weighted)
            sl2, sc2, _ = pair_means(d2, contents, layouts)
            headrooms.append((sc2 / sl2) / ratio - 1.0)
        row = {"spread": float(np.mean(spreads)),
               "headroom": float(np.mean(headrooms)),
               "headroom_sd": float(np.std(headrooms))}
        results["sigma_sweep"][f"sigma={spread}"] = row
        print(f"sigma={spread:4.1f}  spread={row['spread']:.3f}  "
              f"headroom={row['headroom'] * 100:+6.1f}% "
              f"(sd {row['headroom_sd'] * 100:.1f})")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
