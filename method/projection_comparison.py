import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import torch

from factor_eval import (
    cosine_distance,
    layout_differences,
    nearest_neighbour_recall,
    pair_means,
    remove_subspace,
)
from seed_variance import cached_features
from train_projection import Projection


def metrics(matrix, contents, layouts):

    base = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
    distance = cosine_distance(base)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    return {
        "ratio": float(s_content / s_layout) if s_layout > 0 else float("nan"),
        "NN_R@1": float(nearest_neighbour_recall(distance, contents, layouts)),
    }


def main():
    parser = argparse.ArgumentParser(description="Closed form vs trained head.")
    parser.add_argument("--train-grid", default="dataset/sae_train")
    parser.add_argument("--eval-grid", default="survey_grid")
    parser.add_argument("--encoder", default="clip_openai",
                        help="single encoder (kept for compatibility)")
    parser.add_argument("--encoders", default="",
                        help="comma list; if given, every encoder is evaluated and "
                             "the report is keyed by encoder")
    parser.add_argument("--exclude-docs", type=int, default=6)
    parser.add_argument("--ranks", default="2,4,8")
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--linear", default="outputs/trained_projection.pt")
    parser.add_argument("--mlp", default="outputs/trained_projection_mlp.pt")
    parser.add_argument("--diagonal", default="outputs/trained_projection_diagonal.pt")
    parser.add_argument("--heads-encoder", default="clip_openai",
                        help="the encoder the trained heads were trained on; they are "
                             "skipped for any other encoder")
    parser.add_argument("--report-json", default="outputs/projection_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ranks = [int(item) for item in args.ranks.split(",") if item.strip()]
    encoders = ([item.strip() for item in args.encoders.split(",") if item.strip()]
                or [args.encoder])

    if len(encoders) > 1:
        combined = {}
        for encoder in encoders:
            print(f"\n=== {encoder}")
            combined[encoder] = compare(encoder, ranks, args, device)
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(combined, handle, ensure_ascii=False, indent=2)
        print(f"\nreport json: {args.report_json}")
        return

    rows = compare(encoders[0], ranks, args, device)
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


def compare(encoder, ranks, args, device):

    train_features, train_contents, train_layouts = cached_features(
        args.train_grid, "sae_train", encoder, device, args.cache_dir
    )
    keep = [
        index for index, document in enumerate(train_contents)
        if int(document.split("::")[1]) >= args.exclude_docs
    ]
    differences = layout_differences(
        train_features[keep],
        [train_contents[index] for index in keep],
        [train_layouts[index] for index in keep],
        n_pairs=args.pairs,
    )

    torch.manual_seed(0)
    basis = torch.pca_lowrank(differences, q=max(ranks), center=False)[2]
    print(f"closed form fitted on {len(keep)} training images, "
          f"{len(set(train_contents[index] for index in keep))} contents")

    eval_features, eval_contents, eval_layouts = cached_features(
        args.eval_grid, "documents", encoder, device, args.cache_dir
    )

    rows = {"raw": metrics(eval_features, eval_contents, eval_layouts)}
    for rank in ranks:
        reduced = remove_subspace(eval_features, basis[:, :rank])
        rows[f"closed_form_r{rank}"] = metrics(reduced, eval_contents, eval_layouts)

    for name, path in (("trained_linear", args.linear),
                       ("trained_mlp", args.mlp),
                       ("trained_diagonal", args.diagonal)):
        if encoder != args.heads_encoder:
            print(f"  [skip] {name}: trained on {args.heads_encoder}, this is {encoder}")
            continue
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        blob = torch.load(path, weights_only=False)
        config = blob["args"]
        first = next(iter(blob["state_dict"].values()))
        if first.shape[-1] != eval_features.shape[1]:
            print(f"  [skip] {name}: trained on {first.shape[-1]}-d features, "
                  f"this encoder gives {eval_features.shape[1]}")
            continue
        model = Projection(
            eval_features.shape[1],
            config.get("hidden", 0),
            config.get("diagonal", False),
        )
        model.load_state_dict(blob["state_dict"])
        model.eval()
        with torch.no_grad():
            projected = model(eval_features.float())
        rows[name] = metrics(projected, eval_contents, eval_layouts)

    print(f"\n{'variant':22s} {'ratio':>7s} {'NN_R@1':>8s}")
    print("-" * 40)
    for name, row in rows.items():
        print(f"{name:22s} {row['ratio']:7.3f} {row['NN_R@1']:8.3f}")

    return rows


if __name__ == "__main__":
    main()
