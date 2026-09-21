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


def evaluate(distance, contents, layouts, gallery_index, known, unknown):

    gallery_contents = contents[gallery_index]
    ranks = np.argsort(distance[:, gallery_index], axis=1)
    ordered = gallery_contents[ranks]

    def recall_at(k, rows):
        return float(np.mean([contents[i] in ordered[i, :k] for i in rows]))

    def average_precision(i):
        row = ordered[i]
        hits = row == contents[i]
        if not hits.any():
            return 0.0
        positions = np.where(hits)[0]
        return float(np.mean([(np.sum(hits[:p + 1])) / (p + 1)
                              for p in positions]))

    top1 = distance[:, gallery_index].min(axis=1)
    known_rows = np.where(known)[0]
    unknown_rows = np.where(unknown)[0]
    auroc = float("nan")
    if len(known_rows) and len(unknown_rows):
        scores = np.concatenate([-top1[known_rows], -top1[unknown_rows]])
        labels = np.concatenate([np.ones(len(known_rows)), np.zeros(len(unknown_rows))])
        order = np.argsort(scores)
        ranks_ = np.empty(len(scores))
        ranks_[order] = np.arange(1, len(scores) + 1)
        positives = labels == 1
        auroc = float((ranks_[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
                      / (positives.sum() * (~positives).sum()))
    return {
        "R@1": recall_at(1, known_rows),
        "R@5": recall_at(5, known_rows),
        "mAP": float(np.mean([average_precision(i) for i in known_rows])),
        "auroc_known_vs_unknown": auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieval protocol.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--domains", default="documents,arxiv")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--unknown-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-json", default="outputs/retrieval_protocol.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = {}
    for domain in domains:
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            unique = np.asarray(sorted(set(contents.tolist())))
            rng = np.random.default_rng(args.seed)
            unknown = set(rng.permutation(unique)[
                : max(1, int(args.unknown_fraction * len(unique)))].tolist())
            gallery = np.where((layouts == 0) & np.array(
                [c not in unknown for c in contents]))[0]
            known_mask = np.array([c not in unknown for c in contents]) & (layouts != 0)
            unknown_mask = np.array([c in unknown for c in contents])

            def distance_of(matrix):
                base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
                return cosine_distance(base).numpy()

            basis = fit_basis(features, contents, layouts, rank=args.rank)
            row = {
                "raw": evaluate(distance_of(features), contents, layouts,
                                gallery, known_mask, unknown_mask),
                "rotation": evaluate(distance_of(project(features, basis, args.rank)),
                                     contents, layouts, gallery,
                                     known_mask, unknown_mask),
                "gallery": int(len(gallery)),
                "known_queries": int(known_mask.sum()),
                "unknown_queries": int(unknown_mask.sum()),
            }
            results[f"{domain}/{encoder}"] = row
            print(f"{domain:12s} {encoder:12s} "
                  f"raw R@1 {row['raw']['R@1']:.3f} mAP {row['raw']['mAP']:.3f} "
                  f"auroc {row['raw']['auroc_known_vs_unknown']:.3f} | "
                  f"rot R@1 {row['rotation']['R@1']:.3f} mAP {row['rotation']['mAP']:.3f} "
                  f"auroc {row['rotation']['auroc_known_vs_unknown']:.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
