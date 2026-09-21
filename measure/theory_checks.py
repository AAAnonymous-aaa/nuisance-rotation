import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np

from experiment_utils import DOMAINS, load
from factor_eval import variance_decomposition


SURVEYS = {
    "documents": "backbone_survey.json",
    "shapes": "shape_survey.json",
    "real scans": "funsd_scan_survey.json",
    "arxiv": "real_scan_wide_survey.json",
    "real captures": "midv_survey.json",
}


def main():
    parser = argparse.ArgumentParser(description="Theory checks.")
    parser.add_argument("--encoders", default="clip_openai,clip_laion,clip_b32,siglip,"
                                             "clip_datacomp,convnext,eva02_b,dinov2_l")
    parser.add_argument("--report-json", default="outputs/theory_checks.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    rows = []
    for domain, _, _ in DOMAINS:
        if domain not in SURVEYS:
            continue
        survey = json.load(open(os.path.join("outputs", SURVEYS[domain]), encoding="utf-8"))
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)


            if len(features) > 1200:
                unique = sorted(set(contents.tolist()))
                keep = set(unique[:: max(1, len(unique) // 300)])
                mask = np.array([c in keep for c in contents])
                features, contents, layouts = features[mask], contents[mask], layouts[mask]
            layout_var, content_var = variance_decomposition(features, contents, layouts)
            v = np.maximum(layout_var.numpy(), 1e-12)
            c = np.maximum(content_var.numpy(), 1e-12)
            rho = c / v
            rho_bar = float(np.sum(v * rho) / np.sum(v))
            measured = (survey[encoder]["ratio_after_diagonal"] /
                        survey[encoder]["ratio"] - 1.0)


            s_oracle = rho / rho_bar - 1.0
            first_order = float(np.sum(v * s_oracle * rho) / np.sum(v * rho)
                                - np.sum(v * s_oracle) / np.sum(v))
            rows.append({
                "domain": domain,
                "encoder": encoder,
                "cv": float(np.std(rho) / np.mean(rho)),
                "range_ratio": float(rho.max() / rho.min() - 1.0),
                "measured_headroom": measured,
                "max_over_mean": float(rho.max() / rho_bar - 1.0),
                "first_order_prediction": first_order,
            })
            print(f"{domain:14s} {encoder:14s} CV {rows[-1]['cv']:.3f}  "
                  f"range {rows[-1]['range_ratio']:9.1f}  "
                  f"max/mean {rows[-1]['max_over_mean']:8.2f}  "
                  f"1st-order {first_order:6.3f}  measured {measured:6.3f}")

    measured = np.array([r["measured_headroom"] for r in rows])
    first = np.array([r["first_order_prediction"] for r in rows])
    cv = np.array([r["cv"] for r in rows])
    summary = {
        "n": len(rows),
        "corr_cv_measured": float(np.corrcoef(cv, measured)[0, 1]),
        "corr_cv_measured_log": float(np.corrcoef(np.log1p(cv), np.log1p(measured))[0, 1]),
        "first_order_over_measured_median": float(np.median(first / np.maximum(measured, 1e-9))),
        "corr_first_order_measured": float(np.corrcoef(first, measured)[0, 1]),
        "max_over_mean_over_measured_median": float(np.median(
            np.array([r["max_over_mean"] for r in rows]) / np.maximum(measured, 1e-9))),
    }


    controls = json.load(open("outputs/sanity_controls.json", encoding="utf-8"))
    fitted_rows = []
    for key, row in controls.items():
        domain, encoder = key.split("/")
        features, contents, layouts = load(domain, encoder)
        from factor_eval import split_by_content
        left, _ = split_by_content(contents)
        from factor_eval import variance_decomposition as vd
        lv_l, cv_l = vd(features[left], contents[left], layouts[left])
        weights = (cv_l / (lv_l + 1e-12))
        weights = (weights / weights.mean()).clamp(0.1, 10.0)
        lv, cv = vd(features, contents, layouts)
        v = np.maximum(lv.numpy(), 1e-12)
        rho = np.maximum(cv.numpy(), 1e-12) / v
        s = weights.numpy() - 1.0
        predicted = float(np.sum(v * s * rho) / np.sum(v * rho)
                          - np.sum(v * s) / np.sum(v))
        fitted_rows.append({
            "setting": key,
            "first_order_prediction": predicted,
            "measured_operational": row["held_out_oracle_diagonal"]["gain_percent"] / 100.0,
        })
        print(f"fitted weights {key:28s} 1st-order {predicted:+.3f}  "
              f"operational {fitted_rows[-1]['measured_operational']:+.3f}")
    if fitted_rows:
        predicted = np.array([r["first_order_prediction"] for r in fitted_rows])
        operational = np.array([r["measured_operational"] for r in fitted_rows])
        summary["fitted_corr_with_operational"] = float(np.corrcoef(predicted, operational)[0, 1])
        summary["fitted_median_ratio"] = float(np.median(predicted / np.maximum(operational, 1e-9)))
    summary["fitted_rows"] = fitted_rows

    print("\nsummary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summary": summary}, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
