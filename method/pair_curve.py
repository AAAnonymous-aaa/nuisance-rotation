import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from factor_eval import layout_differences, remove_subspace
from experiment_utils import load, metrics


SIZES = (8, 16, 32, 64, 128, 256, 476)
NOISE = (0.0, 0.25, 0.5, 0.75, 1.0)


def mismatched_differences(features, contents, pairs, rng):
    left = rng.integers(0, len(contents), size=pairs)
    right = rng.integers(0, len(contents), size=pairs)
    for _ in range(20):
        bad = contents[left] == contents[right]
        if not bad.any():
            break
        right[bad] = rng.integers(0, len(contents), size=int(bad.sum()))
    keep = contents[left] != contents[right]
    return features[left[keep]] - features[right[keep]]


def main():
    parser = argparse.ArgumentParser(description="Calibration size and pairing noise.")
    parser.add_argument("--train-grid", default="dataset/sae_train")
    parser.add_argument("--train-name", default="sae_train")
    parser.add_argument("--eval-grid", default="survey_grid")
    parser.add_argument("--eval-name", default="documents")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--report-json", default="outputs/pair_curve.json")
    args = parser.parse_args()

    from seed_variance import cached_features

    device = torch.device("cpu")
    train_features, train_contents, train_layouts = cached_features(
        args.train_grid, args.train_name, args.encoder, device, "outputs/cache"
    )
    train_contents = np.asarray(train_contents)
    train_layouts = np.asarray(train_layouts)
    keep = np.array([int(d.split("::")[1]) >= 6 for d in train_contents])
    train_features, train_contents, train_layouts = (
        train_features[keep], train_contents[keep], train_layouts[keep]
    )
    eval_features, eval_contents, eval_layouts = load(args.eval_name, args.encoder)
    unique = np.asarray(sorted(set(train_contents.tolist())))
    print(f"train {tuple(train_features.shape)} ({len(unique)} contents), "
          f"eval {tuple(eval_features.shape)}")

    results = {"baseline": metrics(eval_features, eval_contents, eval_layouts),
               "sizes": {}, "noise": {}}
    print("\ncalibration size")
    for size in SIZES:
        if size > len(unique):
            continue
        rng = np.random.default_rng(0)
        chosen = set(rng.permutation(unique)[:size].tolist())
        mask = np.array([c in chosen for c in train_contents])
        differences = layout_differences(
            train_features[mask], train_contents[mask], train_layouts[mask],
            n_pairs=args.pairs,
        )
        torch.manual_seed(0)
        basis = torch.pca_lowrank(differences, q=args.rank, center=False)[2]
        row = metrics(remove_subspace(eval_features, basis), eval_contents, eval_layouts)
        results["sizes"][size] = row
        print(f"  K={size:4d}  ratio={row['ratio']:5.2f}  R@1={row['NN_R@1']:.3f}")

    print("\npairing noise")
    rng = np.random.default_rng(0)
    matched = layout_differences(
        train_features, train_contents, train_layouts, n_pairs=args.pairs
    )
    mismatched = mismatched_differences(train_features, train_contents,
                                        args.pairs, rng)
    for fraction in NOISE:
        count = int(round(fraction * len(matched)))
        parts = []
        if len(matched) - count:
            parts.append(matched[: len(matched) - count])
        if count:
            parts.append(mismatched[:count])
        differences = torch.cat(parts, dim=0)
        torch.manual_seed(0)
        basis = torch.pca_lowrank(differences, q=args.rank, center=False)[2]
        row = metrics(remove_subspace(eval_features, basis), eval_contents, eval_layouts)
        results["noise"][f"{fraction:.2f}"] = row
        print(f"  mismatched={fraction:4.2f}  ratio={row['ratio']:5.2f}  "
              f"R@1={row['NN_R@1']:.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
