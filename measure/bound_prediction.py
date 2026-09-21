import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np

from experiment_utils import DOMAINS, load
from factor_eval import variance_decomposition


def main():
    parser = argparse.ArgumentParser(description="Predicted vs measured headroom.")
    parser.add_argument("--encoders", default="clip_openai,clip_laion,clip_b32,siglip,"
                                             "clip_datacomp,convnext,eva02_b,dinov2_l")
    parser.add_argument("--report-json", default="outputs/bound_prediction.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    survey_files = {
        "documents": "backbone_survey.json",
        "shapes": "shape_survey.json",
        "real scans": "funsd_scan_survey.json",
        "arxiv": "real_scan_wide_survey.json",
        "real captures": "midv_survey.json",
    }
    rows = []
    for domain, _, _ in DOMAINS:
        if domain not in survey_files:
            continue
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)


            if len(features) > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features = features[mask]
                contents = contents[mask]
                layouts = layouts[mask]
            layout_var, content_var = variance_decomposition(features, contents, layouts)
            v = np.maximum(layout_var.numpy(), 1e-12)
            c = np.maximum(content_var.numpy(), 1e-12)
            rho = c / v
            predicted = float(rho.max() / (np.sum(v * rho) / np.sum(v)) - 1.0)
            measured = None
            data = json.load(open(os.path.join("outputs", survey_files[domain]),
                                  encoding="utf-8"))
            if encoder in data:
                row = data[encoder]
                measured = (row["ratio_after_diagonal"] / row["ratio"] - 1.0)
            if measured is None:
                continue
            rows.append({
                "domain": domain,
                "encoder": encoder,
                "predicted": predicted * 100.0,
                "measured": measured * 100.0,
                "spread": float(np.std(rho) / np.mean(rho)),
            })
            print(f"{domain:14s} {encoder:14s} spread {rows[-1]['spread']:.3f}  "
                  f"predicted {rows[-1]['predicted']:+7.1f}%  "
                  f"measured {rows[-1]['measured']:+6.1f}%")

    predicted = np.array([r["predicted"] for r in rows])
    measured = np.array([r["measured"] for r in rows])
    ratio = predicted / np.maximum(measured, 1e-9)
    summary = {
        "n": len(rows),
        "correlation": float(np.corrcoef(np.log1p(predicted), np.log1p(measured))[0, 1]),
        "median_predicted_over_measured": float(np.median(ratio)),
        "min_predicted_over_measured": float(ratio.min()),
        "max_predicted_over_measured": float(ratio.max()),
        "spearman": float(np.corrcoef(
            np.argsort(np.argsort(predicted)), np.argsort(np.argsort(measured)))[0, 1]),
    }
    print("\npredicted vs measured: "
          f"log-log correlation {summary['correlation']:.3f}, "
          f"median ratio {summary['median_predicted_over_measured']:.2f}, "
          f"range [{summary['min_predicted_over_measured']:.2f}, "
          f"{summary['max_predicted_over_measured']:.2f}]")
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
