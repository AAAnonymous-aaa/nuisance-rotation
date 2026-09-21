import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from diagonal_frontier import SURVEYS, ratio_of, recall
from experiment_utils import DOMAINS, load
from factor_eval import split_by_content, variance_decomposition
from factor_eval import cosine_distance, nearest_neighbour_recall
from backbone_survey import load_any_grid


def as_mask(selection, n):
    selection = np.asarray(selection)
    if selection.dtype == bool:
        return selection
    mask = np.zeros(n, dtype=bool)
    mask[selection] = True
    return mask


def optimise(features, contents, layouts, clip, steps, lr, seed=0):
    torch.manual_seed(seed)
    bound = float(np.log(clip))
    log_weights = torch.zeros(features.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([log_weights], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        value = ratio_of(log_weights.exp(), features, contents, layouts)
        (-value).backward()
        optimizer.step()
        with torch.no_grad():
            log_weights.clamp_(-bound, bound)
    with torch.no_grad():
        return log_weights.exp().detach()


def topic_probe(features, topics, fit_mask, test_mask):
    clf = LogisticRegression(max_iter=300)
    clf.fit(features[fit_mask].numpy(), topics[fit_mask])
    return float(clf.score(features[test_mask].numpy(), topics[test_mask]))


def main():
    parser = argparse.ArgumentParser(description="Validate the reweighting result.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--domains", default="documents,shapes,real scans,large scans,arxiv,identity documents")
    parser.add_argument("--clips", default="2,4")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--report-json", default="outputs/validate_reweighting.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    wanted = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    clips = [float(c) for c in args.clips.split(",") if c.strip()]
    results = {}
    for domain, grid, _ in DOMAINS:
        if domain not in SURVEYS or domain not in wanted:
            continue
        paths, contents_full, _ = load_any_grid(grid)


        topics_full = np.asarray([str(c).split("::")[0] for c in contents_full])
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            topics = topics_full
            n = len(features)
            if n > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features, contents, layouts = features[mask], contents[mask], layouts[mask]
                topics = topics_full[mask]
                n = len(features)
            row = {"n": int(n), "seeds": {}, "realisation_split": {}}
            for seed in range(args.seeds):
                fit_mask = as_mask(split_by_content(contents, seed=seed)[0], n)
                test_mask = ~fit_mask
                raw = float(ratio_of(torch.ones(features.shape[1]), features[test_mask],
                                     contents[test_mask], layouts[test_mask]))
                layout_var, content_var = variance_decomposition(
                    features[fit_mask], contents[fit_mask], layouts[fit_mask])
                plug = content_var / (layout_var + 1e-12)
                plug = plug / plug.mean()
                entry = {
                    "raw_ratio": raw,
                    "raw_recall": recall(torch.ones(features.shape[1]), features[test_mask],
                                         contents[test_mask], layouts[test_mask]),
                    "plugin_clipped": {}, "plugin_unclipped": {}, "optimised": {},
                }
                for name, weights in (("plugin_clipped", plug.clamp(0.1, 10.0).sqrt()),
                                      ("plugin_unclipped", plug.sqrt())):
                    ratio = float(ratio_of(weights, features[test_mask],
                                           contents[test_mask], layouts[test_mask]))
                    entry[name] = {
                        "gain_percent": (ratio / raw - 1.0) * 100.0,
                        "recall": recall(weights, features[test_mask],
                                         contents[test_mask], layouts[test_mask]),
                    }
                for clip in clips:
                    weights = optimise(features[fit_mask], contents[fit_mask],
                                       layouts[fit_mask], clip, args.steps, args.lr, seed)
                    ratio = float(ratio_of(weights, features[test_mask],
                                           contents[test_mask], layouts[test_mask]))
                    entry["optimised"][f"clip={clip:g}"] = {
                        "gain_percent": (ratio / raw - 1.0) * 100.0,
                        "recall": recall(weights, features[test_mask],
                                         contents[test_mask], layouts[test_mask]),
                        "weight_p99_over_median": float(
                            (weights.quantile(0.99) / weights.median()).item()),
                    }
                row["seeds"][f"seed={seed}"] = entry


            realisations = sorted(set(layouts.tolist()))
            fit_mask = np.array([r in realisations[::2] for r in layouts])
            test_mask = ~fit_mask
            raw = float(ratio_of(torch.ones(features.shape[1]), features[test_mask],
                                 contents[test_mask], layouts[test_mask]))
            layout_var, content_var = variance_decomposition(
                features[fit_mask], contents[fit_mask], layouts[fit_mask])
            plug = content_var / (layout_var + 1e-12)
            plug = plug / plug.mean()
            entry = {"raw_ratio": raw}
            for name, weights in (("plugin_clipped", plug.clamp(0.1, 10.0).sqrt()),
                                  ("plugin_unclipped", plug.sqrt())):
                ratio = float(ratio_of(weights, features[test_mask], contents[test_mask],
                                       layouts[test_mask]))
                entry[name] = (ratio / raw - 1.0) * 100.0
            for clip in clips:
                weights = optimise(features[fit_mask], contents[fit_mask],
                                   layouts[fit_mask], clip, args.steps, args.lr, 0)
                ratio = float(ratio_of(weights, features[test_mask], contents[test_mask],
                                       layouts[test_mask]))
                entry[f"optimised clip={clip:g}"] = (ratio / raw - 1.0) * 100.0
            row["realisation_split"] = entry


            fit_mask = as_mask(split_by_content(contents, seed=0)[0], n)
            test_mask = ~fit_mask
            layout_var, content_var = variance_decomposition(
                features[fit_mask], contents[fit_mask], layouts[fit_mask])
            plug = content_var / (layout_var + 1e-12)
            plug = (plug / plug.mean()).clamp(0.1, 10.0).sqrt()
            weights = optimise(features[fit_mask], contents[fit_mask], layouts[fit_mask],
                               clips[0], args.steps, args.lr, 0)
            distinct_topics = np.unique(topics)
            if len(np.unique(topics[fit_mask])) < 2:
                probe = {"raw": None, "plugin": None, "optimised": None,
                         "note": "single topic label in this grid"}
            elif len(distinct_topics) == len(set(contents.tolist())):
                probe = {"raw": None, "plugin": None, "optimised": None,
                         "note": "one label per content; this grid has no topic "
                                 "structure to probe"}
            else:
                probe = {
                    "raw": topic_probe(features, topics, fit_mask, test_mask),
                    "plugin": topic_probe(features * plug, topics, fit_mask, test_mask),
                    "optimised": topic_probe(features * weights, topics, fit_mask, test_mask),
                }
            row["topic_probe"] = probe
            results[f"{domain}/{encoder}"] = row

            gains = [s["optimised"][f"clip={clips[0]:g}"]["gain_percent"]
                     for s in row["seeds"].values()]
            recalls = [s["optimised"][f"clip={clips[0]:g}"]["recall"]
                       for s in row["seeds"].values()]
            raw_recalls = [s["raw_recall"] for s in row["seeds"].values()]
            plugin_gains = [s["plugin_clipped"]["gain_percent"] for s in row["seeds"].values()]
            unclipped = [s["plugin_unclipped"]["gain_percent"] for s in row["seeds"].values()]
            probe_text = ("n/a" if probe["raw"] is None else
                          f"{probe['raw']:.3f}/{probe['plugin']:.3f}/{probe['optimised']:.3f}")
            print(f"{domain:18s} {encoder:12s} "
                  f"plugin {np.mean(plugin_gains):+5.1f}%  "
                  f"unclipped {np.mean(unclipped):+5.1f}%  "
                  f"optimised c={clips[0]:g} {np.mean(gains):+5.1f}%+/-{np.std(gains):.1f}  "
                  f"R@1 raw {np.mean(raw_recalls):.3f} -> {np.mean(recalls):.3f}  "
                  f"realisation {row['realisation_split'][f'optimised clip={clips[0]:g}']:+.1f}%  "
                  f"topic probe {probe_text}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"report json: {args.report_json}")


if __name__ == "__main__":
    main()
