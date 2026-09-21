import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import torch

from experiment_utils import load, metrics


GRIDS = {
    "documents": "sae_logits_survey_grid.pt",
    "shapes": "sae_logits_dataset_shape_grid.pt",
    "real scans": "sae_logits_dataset_funsd_scan.pt",
    "arxiv": "sae_logits_dataset_real_scan_wide.pt",
}


def topk(logits, k):
    values, indices = torch.topk(logits, k, dim=1)
    sparse = torch.zeros_like(logits)
    sparse.scatter_(1, indices, torch.relu(values))
    return sparse


def threshold_rule(logits, k):
    positive = torch.relu(logits)
    values = torch.topk(positive, k, dim=1).values
    cut = values[:, -1:]
    return torch.where(positive >= cut, positive, torch.zeros_like(positive))


def nucleus(logits, mass=0.9):
    values = torch.relu(logits)
    total = values.sum(dim=1, keepdim=True).clamp_min(1e-9)
    sorted_values, order = torch.sort(values, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=1) / total
    keep = (cumulative - (sorted_values / total) <= mass) | (cumulative <= mass)
    kept = sorted_values * keep
    out = torch.zeros_like(values)
    out.scatter_(1, order, kept)
    return out


def main():
    parser = argparse.ArgumentParser(description="Selection strategies.")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--report-json", default="outputs/selection_strategies.json")
    args = parser.parse_args()

    results = {}
    for domain, filename in GRIDS.items():
        path = os.path.join(args.cache_dir, filename)
        if not os.path.exists(path):
            print(f"  [skip] {domain}: {path} not found")
            continue
        blob = torch.load(path, weights_only=False)
        logits, released = blob["logits"], blob["released"]
        _, contents, layouts = load(domain, "clip_openai")
        active = int(round(float((released > 0).sum(dim=1).float().mean())))
        row = {
            "mean_active": active,
            "dense": metrics(logits, contents, layouts),
            "global_topk": metrics(released, contents, layouts),
            "per_stream_topk": metrics(topk(logits, active), contents, layouts),
            "threshold": metrics(threshold_rule(logits, active), contents, layouts),
            "nucleus": metrics(nucleus(logits), contents, layouts),
            "mean_kept": {
                "global_topk": float((released > 0).sum(dim=1).float().mean()),
                "per_stream_topk": active,
                "threshold": float((threshold_rule(logits, active) > 0).sum(dim=1).float().mean()),
                "nucleus": float((nucleus(logits) > 0).sum(dim=1).float().mean()),
            },
        }
        results[domain] = row
        kept = row["mean_kept"]
        print(f"{domain:12s} active~{active:2d}  dense {row['dense']['NN_R@1']:.3f}  "
              f"global {row['global_topk']['NN_R@1']:.3f}  "
              f"per-stream {row['per_stream_topk']['NN_R@1']:.3f}  "
              f"threshold {row['threshold']['NN_R@1']:.3f}  "
              f"nucleus {row['nucleus']['NN_R@1']:.3f} "
              f"(kept: {kept['global_topk']:.0f}/{kept['per_stream_topk']:.0f}/"
              f"{kept['nucleus']:.0f})")

    scaled = json.load(open("outputs/domain_sae_large.json", encoding="utf-8"))
    var = float(torch.load("outputs/cache/sae_large_clip_openai.pt",
                           weights_only=False)["features"].var())
    results["in_domain_sae_quality"] = {
        "images": scaled["images"],
        "dead_latent_fraction": scaled["dead_latent_fraction"],
        "final_train_mse": scaled["final_train_mse"],
        "variance_explained": 1.0 - scaled["final_train_mse"] / var,
    }
    print("in-domain SAE: "
          f"{results['in_domain_sae_quality']['variance_explained']:.4f} of the "
          "feature variance explained by the dense reconstruction")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
