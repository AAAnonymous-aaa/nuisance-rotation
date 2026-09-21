import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from augmentation_pairs import augment, encode_augmented
from backbone_survey import load_any_grid
from experiment_utils import DOMAINS, load, metrics, project
from factor_eval import layout_differences, split_by_content


RANK = 4


def subspace(features):
    torch.manual_seed(0)
    return torch.pca_lowrank(features, q=RANK, center=False)[2]


def principal_cosines(basis_a, basis_b):

    values = torch.linalg.svdvals(basis_a.t() @ basis_b)
    return [float(v) for v in values]


def main():
    parser = argparse.ArgumentParser(description="Condition for the fallback.")
    parser.add_argument("--domains", default="documents;identity documents")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/augmentation_condition.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wanted = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    results = {}
    for domain, grid, _ in DOMAINS:
        if domain not in wanted:
            continue
        features, contents, layouts = load(domain, args.encoder)
        paths, _, _ = load_any_grid(grid)
        left, _ = split_by_content(contents)
        index = np.where(left)[0]
        if len(index) > 400:
            index = index[:: max(1, len(index) // 400)]
        differences = layout_differences(features[index], contents[index],
                                         layouts[index], n_pairs=4000)
        basis_true = subspace(differences)
        print(f"{domain}: augmenting {len(index)} images")
        first, second = encode_augmented([paths[i] for i in index], args.encoder, device)
        basis_aug = subspace((first - second).float())
        cosines = principal_cosines(basis_true, basis_aug)
        fitted_true = project(features, basis_true, RANK)
        fitted_aug = project(features, basis_aug, RANK)
        row = {
            "principal_cosines": cosines,
            "max_cosine": max(cosines),
            "mean_cosine": float(np.mean(cosines)),
            "raw_recall": metrics(features, contents, layouts)["NN_R@1"],
            "true_basis_recall": metrics(fitted_true, contents, layouts)["NN_R@1"],
            "augmentation_basis_recall": metrics(fitted_aug, contents, layouts)["NN_R@1"],
        }
        results[f"{domain}/{args.encoder}"] = row
        print(f"  principal cosines {['%.2f' % c for c in cosines]}  "
              f"recall raw {row['raw_recall']:.3f} -> true {row['true_basis_recall']:.3f}, "
              f"augmentation {row['augmentation_basis_recall']:.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
