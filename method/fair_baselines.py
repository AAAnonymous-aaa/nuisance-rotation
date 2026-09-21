import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from experiment_utils import fit_basis, load, metrics, project
from factor_eval import remove_subspace, split_by_content


def half_split(contents):
    left, right = split_by_content(contents)
    def as_mask(selection):
        selection = np.asarray(selection)
        if selection.dtype == bool:
            return selection
        mask = np.zeros(len(contents), dtype=bool)
        mask[selection] = True
        return mask
    return as_mask(left), as_mask(right)


def idf_weights(features):

    active = (features.abs() > 1e-6).float()
    frequency = active.mean(dim=0).clamp_min(1.0 / len(features))
    weights = torch.log(1.0 / frequency)
    return weights / weights.mean()


def stable_mask(features, layouts, keep=0.25):


    unique = sorted(set(layouts.tolist()))
    means = torch.stack([features[layouts == l].mean(dim=0) for l in unique])
    spread = means.std(dim=0)
    threshold = torch.quantile(spread, 1.0 - keep)
    mask = (spread <= threshold).float()
    return mask


def pca_removal(features, rank, left):
    centred = features[left] - features[left].mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    basis = torch.pca_lowrank(centred, q=rank, center=False)[2]
    return remove_subspace(features, basis)


def mean_match(features, layouts, left):

    out = features.clone()
    overall = features[torch.tensor(np.where(left)[0])].mean(dim=0, keepdim=True)
    for layout in sorted(set(layouts.tolist())):
        index = torch.tensor(np.where(layouts == layout)[0])
        reference = features[index[left[index]]]
        centre = reference.mean(dim=0, keepdim=True)
        out[index] = features[index] - centre + overall
    return out


def ridge_residualise(features, layouts, left, ridge=1e-2):

    unique = sorted(set(layouts.tolist()))
    index_of = {l: i for i, l in enumerate(unique)}
    y = torch.zeros(len(features), len(unique))
    for i, l in enumerate(layouts):
        y[i, index_of[l]] = 1.0
    x = features[left]
    target = y[torch.tensor(np.where(left)[0])]
    gram = x.t() @ x + ridge * len(x) * torch.eye(x.shape[1])
    weights = torch.linalg.solve(gram, x.t() @ target)

    basis = torch.linalg.svd(weights, full_matrices=False)[0][:, :len(unique)]
    return remove_subspace(features, basis)


def probe_accuracy(matrix, contents, layouts):
    left, _ = half_split(contents)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(matrix[left].numpy(), layouts[left])
    predicted = clf.predict(matrix[~left].numpy())
    classes = np.unique(layouts[left])
    return float(np.mean([np.mean(predicted[layouts[~left] == c] == c)
                          for c in classes if (layouts[~left] == c).any()]))


def main():
    parser = argparse.ArgumentParser(description="Fair baseline comparison.")
    parser.add_argument("--domains", default="documents;real scans")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--report-json", default="outputs/fair_baselines.json")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    results = {}
    for domain in domains:
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            left, _ = half_split(contents)
            basis_pairs = fit_basis(features, contents, layouts, rank=args.rank)
            index_left = np.where(left)[0]
            unique = sorted(set(layouts.tolist()))
            labels_onehot = torch.zeros(len(features), len(unique))
            for i, l in enumerate(layouts):
                labels_onehot[i, unique.index(l)] = 1.0
            supervised_basis = torch.pca_lowrank(
                features[index_left] - features[index_left].mean(dim=0, keepdim=True),
                q=args.rank, center=False)[2]

            methods = {
                "raw": features,
                "IDF weighting": features * idf_weights(features),
                "stable mask (25%)": features * stable_mask(features, layouts),
                "PCA removal (no labels, no pairs)": pca_removal(features, args.rank, left),
                "mean matching (labels)": mean_match(features, layouts, left),
                "ridge residualisation (labels)": ridge_residualise(features, layouts, left),
                "label-PCA removal (labels)": remove_subspace(features, supervised_basis),
                "paired-difference PCA (pairs, ours)": project(features, basis_pairs, args.rank),
            }
            row = {}
            print(f"\n{domain} / {encoder}")
            for name, matrix in methods.items():
                stats = metrics(matrix, contents, layouts)
                stats["probe_accuracy"] = probe_accuracy(matrix, contents, layouts)
                row[name] = stats
                print(f"  {name:38s} ratio {stats['ratio']:5.2f}  "
                      f"R@1 {stats['NN_R@1']:.3f}  probe {stats['probe_accuracy']:.3f}")
            results[f"{domain}/{encoder}"] = row

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
