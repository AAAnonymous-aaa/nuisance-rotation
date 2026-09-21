import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import numpy as np
import torch

from factor_eval import (
    cosine_distance,
    layout_differences,
    nearest_neighbour_recall,
    pair_means,
    remove_subspace,
)
from seed_variance import cached_features


DOMAINS = (
    ("documents", "survey_grid", "documents"),
    ("shapes", "dataset/shape_grid", "shapes"),
    ("real scans", "dataset/funsd_scan", "real scans funsd"),
    ("arxiv", "dataset/real_scan_wide", "real scans wide"),
)

RANK = 4


def metrics(matrix, contents, layouts):
    base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
    distance = cosine_distance(base)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    return {
        "ratio": float(s_content / s_layout) if s_layout > 0 else float("nan"),
        "NN_R@1": float(nearest_neighbour_recall(distance, contents, layouts)),
    }


def half_of(contents, seed=0):
    unique = sorted(set(contents))
    rng = np.random.default_rng(seed)
    keep = set(np.asarray(unique)[rng.permutation(len(unique))[: len(unique) // 2]].tolist())
    return np.array([c in keep for c in contents])


def mismatched_differences(matrix, contents, n_pairs=4000, seed=0):

    rng = np.random.default_rng(seed)
    contents = np.asarray(contents)
    left = rng.integers(0, len(contents), size=n_pairs)
    right = rng.integers(0, len(contents), size=n_pairs)
    bad = contents[left] == contents[right]
    tries = 0
    while bad.any() and tries < 20:
        right[bad] = rng.integers(0, len(contents), size=int(bad.sum()))
        bad = contents[left] == contents[right]
        tries += 1
    keep = ~bad
    return matrix[left[keep]] - matrix[right[keep]]


def basis_from(differences, rank=RANK, seed=0):
    torch.manual_seed(seed)
    return torch.pca_lowrank(differences, q=rank, center=False)[2]


def main():
    parser = argparse.ArgumentParser(description="Subspace controls.")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/subspace_controls.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = {}
    for name, grid, cache_name in DOMAINS:
        features, contents, layouts = cached_features(
            grid, cache_name, args.encoder, device, args.cache_dir
        )
        data[name] = (features, contents, layouts)
        print(f"loaded {name:11s} {tuple(features.shape)}")

    results = {"rank": RANK, "baselines": {}, "control": {}, "transfer": {}}
    for name, (features, contents, layouts) in data.items():
        results["baselines"][name] = metrics(features, contents, layouts)

    print("\nnegative control: fit on mismatched (different-content) pairs")
    features, contents, layouts = data["documents"]
    mask = half_of(contents)
    matched = layout_differences(features[mask], np.asarray(contents)[mask],
                                 np.asarray(layouts)[mask], n_pairs=4000)
    mismatched = mismatched_differences(features[mask], np.asarray(contents)[mask])
    for label, differences in (("matched pairs", matched), ("mismatched pairs", mismatched)):
        basis = basis_from(differences)
        row = metrics(remove_subspace(features, basis), contents, layouts)
        results["control"][label] = row
        print(f"  {label:18s} ratio={row['ratio']:5.2f} R@1={row['NN_R@1']:.3f}")
    results["control"]["no removal"] = results["baselines"]["documents"]
    print(f"  {'no removal':18s} ratio={results['baselines']['documents']['ratio']:5.2f} "
          f"R@1={results['baselines']['documents']['NN_R@1']:.3f}")

    print("\ntransfer: fit on the row domain, evaluate on the column domain")
    names = [name for name, _, _ in DOMAINS]
    print(f"{'fit \\ eval':12s}" + "".join(f"{n:>22s}" for n in names))
    for fit_name in names:
        fit_features, fit_contents, fit_layouts = data[fit_name]
        mask = half_of(fit_contents)
        differences = layout_differences(
            fit_features[mask], np.asarray(fit_contents)[mask],
            np.asarray(fit_layouts)[mask], n_pairs=4000
        )
        basis = basis_from(differences)
        row_out = {}
        line = f"{fit_name:12s}"
        for eval_name in names:
            features, contents, layouts = data[eval_name]
            row = metrics(remove_subspace(features, basis), contents, layouts)
            base = results["baselines"][eval_name]
            row_out[eval_name] = row
            line += f"  {row['ratio']:6.2f}/{row['NN_R@1']:.3f}"
        results["transfer"][fit_name] = row_out
        print(line)
    line = f"{'raw':12s}"
    for eval_name in names:
        base = results["baselines"][eval_name]
        line += f"  {base['ratio']:6.2f}/{base['NN_R@1']:.3f}"
    print(line + "   (ratio / NN_R@1)")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
