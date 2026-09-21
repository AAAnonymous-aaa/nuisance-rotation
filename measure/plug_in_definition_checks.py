import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch

from diagonal_frontier import SURVEYS, optimise, ratio_of
from experiment_utils import DOMAINS, load
from factor_eval import split_by_content, variance_decomposition


def as_mask(selection, n):
    selection = np.asarray(selection)
    if selection.dtype == bool:
        return selection
    mask = np.zeros(n, dtype=bool)
    mask[selection] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description="Plug-in definition checks.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--report-json", default="outputs/plug_in_checks.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in SURVEYS:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            n = len(features)
            if n > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features, contents, layouts = features[mask], contents[mask], layouts[mask]
                n = len(features)
            fit = as_mask(split_by_content(contents, seed=0)[0], n)
            test = ~fit

            layout_var, content_var = variance_decomposition(
                features[fit], contents[fit], layouts[fit])
            plug = content_var / (layout_var + 1e-12)
            plug = plug / plug.mean()
            profile = plug.numpy()
            fitted = optimise(features[fit], contents[fit], layouts[fit],
                              args.clip, args.steps, 0.05, 0).numpy()
            fitted = fitted / fitted.mean()

            corr_plug = float(np.corrcoef(fitted, np.sqrt(profile))[0, 1])
            corr_log = float(np.corrcoef(np.log(fitted), 0.5 * np.log(profile))[0, 1])


            weighted = features[test] * torch.tensor(fitted, dtype=torch.float32)
            norms = weighted.norm(dim=-1)
            raw_norms = features[test].norm(dim=-1)
            norm_variation = float((norms / norms.mean()).std())
            raw_norm_variation = float((raw_norms / raw_norms.mean()).std())

            raw_ratio = float(ratio_of(torch.ones(features.shape[1]), features[test],
                                       contents[test], layouts[test]))
            fitted_ratio = float(ratio_of(torch.tensor(fitted, dtype=torch.float32),
                                          features[test], contents[test], layouts[test]))
            results[f"{domain}/{encoder}"] = {
                "corr_fitted_vs_plug_in": corr_plug,
                "corr_log_fitted_vs_log_plug_in": corr_log,
                "norm_variation_raw": raw_norm_variation,
                "norm_variation_fitted": norm_variation,
                "raw_ratio": raw_ratio,
                "fitted_ratio": fitted_ratio,
                "fitted_gain_percent": (fitted_ratio / raw_ratio - 1.0) * 100.0,
            }
            print(f"{domain:18s} {encoder:12s} corr(fitted, plug-in) {corr_plug:+5.2f}  "
                  f"corr(log) {corr_log:+5.2f}  norm variation {raw_norm_variation:.3f} -> "
                  f"{norm_variation:.3f}  gain {results[f'{domain}/{encoder}']['fitted_gain_percent']:+.1f}%")

    corrs = [r["corr_fitted_vs_plug_in"] for r in results.values()]
    print(f"\nmedian correlation between fitted and plug-in weights: {np.median(corrs):+.3f}")
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
