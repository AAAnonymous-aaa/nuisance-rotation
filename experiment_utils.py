import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from factor_eval import (
    cosine_distance,
    layout_differences,
    nearest_neighbour_recall,
    nearest_neighbour_recall_full,
    pair_means,
    remove_subspace,
)
from seed_variance import cached_features


DOMAINS = (
    ("documents", "survey_grid", "documents"),
    ("shapes", "dataset/shape_grid", "shapes"),
    ("real scans", "dataset/funsd_scan", "real scans funsd"),
    ("large scans", "dataset/docvqa_scan", "docvqa_scan"),
    ("arxiv", "dataset/real_scan_wide", "real scans wide"),
    ("identity documents", "dataset/midv_full", "midv_full"),
    ("real captures", "dataset/midv_capture", "midv_capture"),
)

ENCODERS = ("clip_openai", "clip_laion", "clip_b32", "siglip",
            "clip_datacomp", "convnext", "eva02_b", "dinov2_l")


def load(domain, encoder, cache_dir="outputs/cache", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, grid, cache_name in DOMAINS:
        if name == domain:
            features, contents, layouts = cached_features(
                grid, cache_name, encoder,
                torch.device(device), cache_dir,
            )
            return features, np.asarray(contents), np.asarray(layouts)
    raise KeyError(
        f"unknown domain {domain!r}; valid domains are "
        f"{[name for name, _, _ in DOMAINS]}"
    )


def metrics(matrix, contents, layouts, full=False):
    base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
    distance = cosine_distance(base)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    row = {
        "ratio": float(s_content / s_layout) if s_layout > 0 else float("nan"),
        "NN_R@1": float(nearest_neighbour_recall(distance, contents, layouts)),
    }
    if full:
        row["NN_full"] = float(nearest_neighbour_recall_full(distance, contents, layouts))
    return row


def half_of(contents, seed=0):
    unique = sorted(set(contents.tolist()))
    rng = np.random.default_rng(seed)
    keep = set(np.asarray(unique)[rng.permutation(len(unique))[: len(unique) // 2]].tolist())
    return np.array([c in keep for c in contents])


def fit_basis(features, contents, layouts, rank, mask=None, pairs=4000, seed=0):
    if mask is None:
        mask = half_of(contents)
    differences = layout_differences(
        features[mask], contents[mask], layouts[mask], n_pairs=pairs
    )
    if differences is None or differences.shape[0] < 2:
        raise ValueError(
            "not enough same-content pairs to fit a subspace; the grid needs at "
            "least one content with two renderings"
        )
    torch.manual_seed(seed)
    q = int(min(rank, differences.shape[0] - 1, differences.shape[1]))
    return torch.pca_lowrank(differences, q=max(1, q), center=False)[2]


def project(features, basis, rank):
    if rank <= 0:
        return features
    return remove_subspace(features, basis[:, :rank])
