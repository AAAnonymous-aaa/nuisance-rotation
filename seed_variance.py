import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from backbone_survey import OPEN_CLIP_MODELS, analyse, load_any_grid, load_open_clip
from backbone_survey import DINO_MODELS, dino_transform, load_dino
from train_domain_sae import SparseAutoencoder


GRIDS = {
    "documents": "survey_grid",
    "shapes": "dataset/shape_grid",
    "real scans wide": "dataset/real_scan_wide",
    "real scans funsd": "dataset/funsd_scan",
    "real captures": "dataset/midv_capture",
}


@torch.no_grad()
def encode(paths, model, transform, device, batch_size=16):
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="encode", leave=False):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        tensor = torch.stack([transform(image) for image in images]).to(device)
        vector = model.encode_image(tensor).float()
        chunks.append((vector / vector.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(chunks, dim=0)


def cached_features(grid_path, grid_name, encoder, device, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{grid_name}_{encoder}.pt".replace(" ", "_"))
    paths, contents, layouts = load_any_grid(grid_path)
    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=False)
        return blob["features"], contents, layouts
    if encoder in DINO_MODELS:
        model = load_dino(DINO_MODELS[encoder], device)
        transform = dino_transform()
        features = []
        with torch.no_grad():
            for start in tqdm(range(0, len(paths), 16), desc="encode", leave=False):
                batch = paths[start : start + 16]
                images = [Image.open(path).convert("RGB") for path in batch]
                tensor = torch.stack([transform(image) for image in images]).to(device)
                features.append(model(tensor).float().cpu())
        features = torch.cat(features, dim=0)
    else:
        model_name, tag, _ = OPEN_CLIP_MODELS[encoder]
        model, transform = load_open_clip(model_name, tag, device)
        features = encode(paths, model, transform, device)
    torch.save({"features": features, "contents": contents, "layouts": layouts}, cache)
    return features, contents, layouts


def summarise(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std())


def main():
    parser = argparse.ArgumentParser(description="Resampling error bars.")
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--encoders",
                        default="clip_openai,clip_laion,clip_b32,siglip,"
                                "clip_datacomp,convnext,eva02_b,dinov2_l")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--report-json", default="outputs/seed_variance.json")
    parser.add_argument("--include-domain-sae", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="in-domain SAE heads for the document grid")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoders = [item.strip() for item in args.encoders.split(",") if item.strip()]

    results = {}
    for grid_name, grid_path in GRIDS.items():
        for encoder in encoders:
            try:
                features, contents, layouts = cached_features(
                    grid_path, grid_name, encoder, device, args.cache_dir
                )
            except Exception as exc:
                print(f"  [fail] {grid_name}/{encoder}: {type(exc).__name__}")
                continue
            variants = {encoder: features}
            if args.include_domain_sae and grid_name == "documents" and encoder == "clip_openai":
                sae_path = "outputs/domain_sae.pt"
                if os.path.exists(sae_path):
                    blob = torch.load(sae_path, weights_only=False)
                    config = blob["args"]
                    sae = SparseAutoencoder(
                        features.shape[1], config["n_latents"], config["k"],
                        config.get("auxk") or 0,
                    ).to(device)
                    sae.load_state_dict(blob["state_dict"])
                    sae.eval()
                    with torch.no_grad():
                        logits, sparse, _, _ = sae(features.to(device))
                    variants["domain_sae_logits"] = logits.cpu()
                    variants["domain_sae_sparse"] = sparse.cpu()

            contents = np.asarray(contents)
            layouts = np.asarray(layouts)
            unique = sorted(set(contents.tolist()))
            rng = np.random.default_rng(0)
            for name, matrix in variants.items():
                spreads, diagonals, ratios, projections = [], [], [], []
                for _ in range(args.splits):
                    count = max(4, int(len(unique) * args.fraction))
                    chosen = set(
                        np.asarray(unique)[
                            rng.choice(len(unique), count, replace=False)
                        ].tolist()
                    )
                    index = np.array(
                        [i for i, c in enumerate(contents) if c in chosen]
                    )
                    sub_features = matrix[index]
                    sub_contents = contents[index].tolist()
                    sub_layouts = layouts[index].tolist()
                    row = analyse(sub_features, sub_contents, sub_layouts)
                    spreads.append(row["per_dim_ratio_spread"])
                    ratios.append(row["ratio"])
                    diagonals.append(
                        100.0 * (row["ratio_after_diagonal"] / row["ratio"] - 1.0)
                    )
                    if row["ratio_after_rotation"]:
                        projections.append(
                            100.0
                            * (max(row["ratio_after_rotation"].values())
                               / row["ratio"] - 1.0)
                        )
                key = f"{grid_name}/{name}"
                results[key] = {
                    "spread_mean": summarise(spreads)[0],
                    "spread_std": summarise(spreads)[1],
                    "oracle_mean": summarise(diagonals)[0],
                    "oracle_std": summarise(diagonals)[1],
                    "ratio_mean": summarise(ratios)[0],
                    "projection_mean": summarise(projections)[0]
                    if projections else float("nan"),
                    "splits": args.splits,
                }
                print(f"{key:44s} spread={results[key]['spread_mean']:.3f}"
                      f"+/-{results[key]['spread_std']:.3f}  "
                      f"oracle={results[key]['oracle_mean']:+.1f}%"
                      f"+/-{results[key]['oracle_std']:.1f}%")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print()
    print("Summary by domain (mean over encoders, and the spread of oracle gain)")
    for domain in GRIDS:
        rows = [v for k, v in results.items() if k.startswith(domain + "/")]
        if not rows:
            continue
        spread = np.mean([r["spread_mean"] for r in rows])
        oracle = np.mean([r["oracle_mean"] for r in rows])
        between = np.std([r["oracle_mean"] for r in rows])
        within = np.mean([r["oracle_std"] for r in rows])
        print(f"  {domain:12s} spread={spread:.3f}  oracle={oracle:+.1f}%  "
              f"(between-encoder sd {between:.1f}, within-encoder seed sd {within:.1f})")
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
