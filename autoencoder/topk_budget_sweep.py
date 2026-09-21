import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import torch

from backbone_survey import analyse, load_any_grid
from factor_eval import encode_all, load_models


GRIDS = (
    ("documents", "survey_grid"),
    ("shapes", "dataset/shape_grid"),
    ("real scans", "dataset/funsd_scan"),
    ("arxiv", "dataset/real_scan_wide"),
)

BUDGETS = (4, 8, 16, 32, 64, 128, 256, 1024, 0)


def sparse_at(logits, k):

    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k, dim=1)
    sparse = torch.zeros_like(logits)
    sparse.scatter_(1, indices, torch.relu(values))
    return sparse


def main():
    parser = argparse.ArgumentParser(description="Top-k budget sweep.")
    parser.add_argument("--checkpoint-dir", default="model")
    parser.add_argument("--clip-weights",
                        default="clip_weights/open_clip_pytorch_model.bin")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--report-json", default="outputs/topk_budget.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.cache_dir, exist_ok=True)
    models = None

    results = {}
    for domain, grid in GRIDS:
        cache = os.path.join(args.cache_dir, f"sae_logits_{grid.replace('/', '_')}.pt")
        paths, contents, layouts = load_any_grid(grid)
        if os.path.exists(cache):
            blob = torch.load(cache, weights_only=False)
            logits = blob["logits"]
            released = blob["released"]
        else:
            if models is None:
                models = load_models(args.checkpoint_dir, args.clip_weights, device)
            features = encode_all(
                paths,
                ["logits_clip_img", "sparse_codes_clip_img"],
                models,
                device,
                args.batch_size,
            )
            logits = features["sae:logits_clip_img"]
            released = features["sae:sparse_codes_clip_img"]
            torch.save({"logits": logits, "released": released}, cache)

        active = float((released > 0).sum(dim=1).float().mean())
        rows = {}
        for k in BUDGETS:
            code = sparse_at(logits, k)
            stats = analyse(code, contents, layouts)
            name = "dense" if k <= 0 else f"k={k}"
            rows[name] = {
                "ratio": stats["ratio"],
                "NN_R@1": stats["NN_R@1"],
                "spread": stats["per_dim_ratio_spread"],
            }
            print(f"{domain:11s} {name:8s} ratio={stats['ratio']:5.2f} "
                  f"R@1={stats['NN_R@1']:.3f} spread={stats['per_dim_ratio_spread']:.3f}")
        rows["released_mean_active"] = active
        results[domain] = rows
        print(f"{domain:11s} released top-k code: {active:.1f} active CLIP latents of 8192")

    print("\nNN_R@1 by budget")
    header = "domain      " + "".join(f"{('dense' if k <= 0 else 'k=%d' % k):>8s}" for k in BUDGETS)
    print(header)
    for domain, rows in results.items():
        line = f"{domain:11s} " + "".join(
            f"{rows['dense' if k <= 0 else 'k=%d' % k]['NN_R@1']:8.3f}" for k in BUDGETS
        )
        print(line)

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
