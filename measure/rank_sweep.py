import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

from experiment_utils import DOMAINS, fit_basis, load, metrics, project


RANKS = (0, 1, 2, 4, 8, 16, 32)


def main():
    parser = argparse.ArgumentParser(description="Rank sweep for the rotation.")
    parser.add_argument("--encoders", default="clip_openai,siglip,dinov2_l")
    parser.add_argument("--domains", default="documents,shapes,real scans,arxiv,real captures")
    parser.add_argument("--report-json", default="outputs/rank_sweep.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    wanted = [d.strip() for d in args.domains.split(",") if d.strip()]
    results = {}
    for domain, _, _ in DOMAINS:
        if domain not in wanted:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            basis = fit_basis(features, contents, layouts, rank=max(RANKS))
            row = {}
            for rank in RANKS:
                stats = metrics(project(features, basis, rank), contents, layouts)
                row[f"r={rank}"] = stats
            results[f"{domain}/{encoder}"] = row
            line = f"{domain:15s} {encoder:14s}"
            for rank in RANKS:
                line += f"  r{rank}: {row[f'r={rank}']['NN_R@1']:.3f}"
            print(line)

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
