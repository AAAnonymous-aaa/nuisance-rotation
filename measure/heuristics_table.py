import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os


SURVEYS = (
    ("documents", "outputs/backbone_survey.json"),
    ("documents (domain SAE)", "outputs/domain_sae_survey.json"),
    ("shapes", "outputs/shape_survey.json"),
    ("real scans", "outputs/funsd_scan_survey.json"),
    ("large real scans", "outputs/docvqa_scan_survey.json"),
    ("arxiv", "outputs/real_scan_wide_survey.json"),
    ("identity documents", "outputs/midv_full_survey.json"),
)

HEURISTICS = ("idf", "mask", "denoise")


def main():
    parser = argparse.ArgumentParser(description="Heuristics panel.")
    parser.add_argument("--report-json", default="outputs/heuristics_table.json")
    args = parser.parse_args()

    table = {}
    print(f"{'domain':24s} {'n':>3s} " + " ".join(f"{h:>16s}" for h in HEURISTICS))
    for domain, path in SURVEYS:
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        rows = {}
        for encoder, row in data.items():
            if "ratio_after_heuristics" not in row:
                continue
            base = row["ratio"]
            rows[encoder] = {
                h: (row["ratio_after_heuristics"][h] / base - 1.0) * 100.0
                for h in HEURISTICS
                if h in row["ratio_after_heuristics"]
            }
        if not rows:
            continue
        table[domain] = rows
        summary = {}
        for h in HEURISTICS:
            values = [r[h] for r in rows.values() if h in r]
            summary[h] = (min(values), max(values)) if values else (float("nan"),) * 2
        print(f"{domain:24s} {len(rows):3d} " +
              " ".join(f"{summary[h][0]:+6.1f}..{summary[h][1]:+6.1f}" for h in HEURISTICS))

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
