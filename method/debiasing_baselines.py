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
from sklearn.linear_model import LogisticRegression

from factor_eval import (
    cosine_distance,
    layout_differences,
    nearest_neighbour_recall,
    pair_means,
    remove_subspace,
)
from seed_variance import cached_features


GRIDS = (
    ("documents", "survey_grid", "documents"),
    ("shapes", "dataset/shape_grid", "shapes"),
    ("real scans", "dataset/funsd_scan", "real scans funsd"),
    ("arxiv", "dataset/real_scan_wide", "real scans wide"),
)

ENC = ("clip_openai", "clip_laion", "clip_b32", "siglip",
       "clip_datacomp", "convnext", "eva02_b", "dinov2_l")

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


def leace_fit(features, labels, rank=RANK, ridge=1e-3):

    x = features.double()
    mean = x.mean(dim=0, keepdim=True)
    centred = x - mean
    covariance = centred.t() @ centred / max(1, len(x) - 1)
    covariance = covariance + (
        ridge * torch.eye(x.shape[1], dtype=x.dtype) * covariance.diagonal().mean()
    )
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(1e-8)
    whitener = vectors @ torch.diag(values.rsqrt()) @ vectors.t()
    unwhitener = vectors @ torch.diag(values.sqrt()) @ vectors.t()

    whitened = centred @ whitener
    classes = sorted(set(labels))
    means = torch.stack([whitened[torch.tensor([l == c for l in labels])].mean(dim=0)
                         for c in classes])
    centred_means = means - means.mean(dim=0, keepdim=True)
    basis = torch.linalg.svd(centred_means, full_matrices=False)[2][:rank]

    def transform(other):
        y = other.double() - mean
        w = y @ whitener
        return ((w - (w @ basis.t()) @ basis) @ unwhitener).float()

    return transform


def inlp_fit(features, labels, rank=RANK, max_iter=300):

    current = features.float().clone()
    directions = []
    y = np.asarray(labels)
    for _ in range(rank):
        x = current.numpy()
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        classifier = LogisticRegression(max_iter=max_iter, C=1.0)
        try:
            classifier.fit(x, y)
        except Exception:
            break
        weights = torch.tensor(np.atleast_2d(classifier.coef_), dtype=torch.float32)
        direction = weights.mean(dim=0, keepdim=True)
        direction = direction / (direction.norm() + 1e-9)
        directions.append(direction)
        current = current - (current @ direction.t()) @ direction

    def transform(other):
        z = other.float()
        for direction in directions:
            z = z - (z @ direction.t()) @ direction
        return z

    return transform


def main():
    parser = argparse.ArgumentParser(description="Debiasing baselines.")
    parser.add_argument("--encoders", default=",".join(ENC))
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/debiasing_baselines.json")
    args = parser.parse_args()

    device = torch.device(args.device or "cpu")
    encoders = [item.strip() for item in args.encoders.split(",") if item.strip()]

    results = {}
    summary = {name: {"wins": 0, "n": 0, "ratio": 0.0, "recall": 0.0}
               for name in ("leace_all", "leace_half", "inlp_all", "inlp_half",
                            "rotation")}
    for domain, grid, cache_name in GRIDS:
        for encoder in encoders:
            features, contents, layouts = cached_features(
                grid, cache_name, encoder, device, args.cache_dir
            )
            contents = np.asarray(contents)
            layouts = np.asarray(layouts)
            mask = half_of(contents)
            differences = layout_differences(
                features[mask], contents[mask], layouts[mask], n_pairs=4000
            )
            torch.manual_seed(0)
            basis = torch.pca_lowrank(differences, q=RANK, center=False)[2]

            row = {
                "raw": metrics(features, contents, layouts),
                "leace_all": metrics(
                    leace_fit(features, layouts.tolist())(features), contents, layouts),
                "leace_half": metrics(
                    leace_fit(features[mask], layouts[mask].tolist())(features),
                    contents, layouts),
                "inlp_all": metrics(
                    inlp_fit(features, layouts.tolist(), rank=20)(features),
                    contents, layouts),
                "inlp_half": metrics(
                    inlp_fit(features[mask], layouts[mask].tolist(), rank=20)(features),
                    contents, layouts),
                "rotation": metrics(remove_subspace(features, basis), contents, layouts),
            }
            results[f"{domain}/{encoder}"] = row
            base = row["raw"]
            line = f"{domain:11s} {encoder:14s} raw {base['ratio']:5.2f}/{base['NN_R@1']:.3f}"
            for name in ("leace_all", "leace_half", "inlp_all", "inlp_half", "rotation"):
                gain = (row[name]["NN_R@1"] - base["NN_R@1"]) * 100
                line += f"  {name} {row[name]['ratio']:5.2f}/{row[name]['NN_R@1']:.3f} ({gain:+.1f})"
                summary[name]["n"] += 1
                summary[name]["wins"] += int(row[name]["NN_R@1"] > base["NN_R@1"])
                summary[name]["ratio"] += row[name]["ratio"] - base["ratio"]
                summary[name]["recall"] += gain
            print(line)

    print("\nmean change over settings (ratio, R@1 points) and win rate vs raw")
    for name, stats in summary.items():
        print(f"  {name:9s} d_ratio {stats['ratio'] / stats['n']:+6.3f}  "
              f"d_R@1 {stats['recall'] / stats['n']:+5.2f}  "
              f"wins {stats['wins']}/{stats['n']}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": results, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
