import os


if os.environ.get("SURVEY_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import math
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from factor_eval import (
    cosine_distance,
    idf_weights,
    layout_differences,
    load_grid,
    nearest_neighbour_recall,
    pair_means,
    remove_subspace,
    split_by_content,
    variance_decomposition,
)


OPEN_CLIP_MODELS = {
    "clip_openai": ("ViT-L-14", "openai", 224),
    "clip_laion": ("ViT-L-14", "laion2b_s32b_b82k", 224),
    "clip_b32": ("ViT-B-32", "laion2b_e16", 224),
    "clip_b16": ("ViT-B-16", "laion2b_s34b_b88k", 224),
    "siglip": ("ViT-B-16-SigLIP", "webli", 224),
    "clip_datacomp": ("ViT-L-14", "datacomp_xl_s13b_b90k", 224),
    "convnext": ("convnext_base_w", "laion2b_s13b_b82k", 224),
    "eva02_b": ("EVA02-B-16", "merged2b_s8b_b131k", 224),
}

DINO_MODELS = {
    "dinov2_l": "dinov2_vitl14_reg",
    "dinov2_b": "dinov2_vitb14_reg",
}


def load_open_clip(model_name, pretrained, device):

    import open_clip

    model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, transform


def load_dino(name, device):
    model = torch.hub.load("facebookresearch/dinov2", name).to(device)
    model.eval()
    return model


@torch.no_grad()
def encode(paths, encoder, device, batch_size):
    kind, payload = encoder
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc=kind, leave=False):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        if kind == "openclip":
            model, transform = payload
            tensor = torch.stack([transform(image) for image in images]).to(device)
            features = model.encode_image(tensor).float()
        else:
            model, transform = payload
            tensor = torch.stack([transform(image) for image in images]).to(device)
            features = model(tensor).float()
        chunks.append(features.cpu())
    return torch.cat(chunks, dim=0)


def dino_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def normalise(matrix):
    return matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)


def ratio_of(matrix, contents, layouts):
    distance = cosine_distance(matrix)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    return s_content / s_layout if s_layout > 0 else float("nan")


def fit_whitener(features, q=128, eps=1e-6):


    features = features.float()
    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean
    q = int(min(q, min(centred.shape) - 1))
    if q < 2:
        return {"mean": mean, "directions": None}
    _, singular, directions = torch.pca_lowrank(centred, q=q, center=False)
    eigenvalues = (singular ** 2) / max(1, centred.shape[0] - 1)
    total = float((centred ** 2).sum(dim=1).mean())
    remainder = max(total - float(eigenvalues.sum()), 0.0)
    residual = remainder / max(1, features.shape[1] - directions.shape[1])
    return {
        "mean": mean,
        "directions": directions,
        "scale": 1.0 / torch.sqrt(eigenvalues + eps),
        "residual_scale": 1.0 / math.sqrt(residual + eps),
    }


def apply_whitener(features, fitted):
    if fitted.get("directions") is None:
        return features
    centred = features.float() - fitted["mean"]
    directions = fitted["directions"]
    projected = centred @ directions
    inside = (projected * fitted["scale"]) @ directions.T
    outside = centred - projected @ directions.T
    return inside + outside * fitted["residual_scale"]


def heuristic_variants(base, contents, layouts, mask_ratio=0.25):


    gallery_rows = [index for index, layout in enumerate(layouts) if layout == 0]
    gallery = base[gallery_rows]
    out = {}

    weights = idf_weights(gallery)
    out["idf"] = normalise(base * weights)

    active_fraction = (gallery > 0).float().mean(dim=0)
    keep = (active_fraction >= mask_ratio).float()
    out["mask"] = normalise(base * keep)


    rows = []
    for row in base:
        positive = row[row > 0]
        if positive.numel() == 0:
            rows.append(torch.zeros_like(row))
            continue
        threshold = torch.quantile(positive, 0.99) * 0.01
        rows.append(torch.where(row > threshold, row, torch.zeros_like(row)))
    out["denoise"] = normalise(torch.stack(rows, dim=0))
    return out


def load_any_grid(root):


    layout_manifest = os.path.join(root, "layout_manifest.json")
    topic_manifest = os.path.join(root, "manifest.json")
    if os.path.exists(layout_manifest):
        return load_grid(root)
    if not os.path.exists(topic_manifest):
        raise FileNotFoundError(
            f"no layout_manifest.json or manifest.json under {root}"
        )
    with open(topic_manifest, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    paths = [record["path"] for record in records]
    contents = [record.get("document", record.get("sample_id")) for record in records]
    layouts = [record["layout"] for record in records]
    print(
        f"topic manifest: {len(paths)} images, {len(set(contents))} contents, "
        f"{len(set(layouts))} layouts"
    )
    return paths, contents, layouts


def analyse(matrix, contents, layouts, rotation_ranks=(4, 16)):
    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    base = normalise(matrix)
    distance = cosine_distance(base)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    ratio = s_content / s_layout if s_layout > 0 else float("nan")
    recall = nearest_neighbour_recall(distance, contents, layouts)

    layout_var, content_var = variance_decomposition(matrix, contents, layouts)
    a, b = content_var.numpy(), layout_var.numpy()
    uniformity = (
        float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    )


    per_dim = a / (b + 1e-12)
    spread = float(per_dim.std() / (per_dim.mean() + 1e-12))
    median_ratio = float(np.median(per_dim))


    weights = content_var / (layout_var + 1e-12)
    weights = weights / (weights.mean() + 1e-12)
    weights = weights.clamp(0.1, 10.0).sqrt()
    weighted = normalise(base * weights)
    d2 = cosine_distance(weighted)
    sl2, sc2, _ = pair_means(d2, contents, layouts)
    ratio_diag = sc2 / sl2 if sl2 > 0 else float("nan")


    left, right = split_by_content(contents)
    differences = layout_differences(matrix[left], contents[left], layouts[left], n_pairs=4000)
    ratio_rot = {}
    recall_rot = {}
    if differences is not None and differences.shape[0] > 1:
        top = min(max(rotation_ranks), differences.shape[1], differences.shape[0] - 1)
        basis = torch.pca_lowrank(differences, q=max(2, top), center=False)[2]
        recall_before = nearest_neighbour_recall(
            cosine_distance(base[right]), contents[right], layouts[right]
        )
        for rank in rotation_ranks:
            reduced = normalise(remove_subspace(base[right], basis[:, :rank]))
            d3 = cosine_distance(reduced)
            sl3, sc3, _ = pair_means(d3, contents[right], layouts[right])
            ratio_rot[rank] = sc3 / sl3 if sl3 > 0 else float("nan")
            recall_rot[rank] = nearest_neighbour_recall(
                d3, contents[right], layouts[right]
            )
        recall_rot["before"] = recall_before

    heuristics = {
        name: ratio_of(variant, contents, layouts)
        for name, variant in heuristic_variants(base, contents, layouts).items()
    }


    methods = {}
    gallery_left = [index for index in left if layouts[index] == 0]
    if gallery_left:
        gallery = base[gallery_left]
        centre_mean = gallery.mean(dim=0, keepdim=True)
        centred = normalise(base[right] - centre_mean)
        methods["center"] = {
            "ratio": ratio_of(centred, contents[right], layouts[right]),
            "NN_R@1": nearest_neighbour_recall(
                cosine_distance(centred), contents[right], layouts[right]
            ),
        }
        for label, fit_data in (
            ("whiten_gallery", gallery),
            ("whiten_nuisance", None),
        ):
            if label == "whiten_nuisance":
                fit_data = layout_differences(
                    base[left], contents[left], layouts[left], n_pairs=4000
                )
            if fit_data is None or fit_data.shape[0] < 2:
                continue
            fitted = fit_whitener(fit_data)
            transformed = normalise(apply_whitener(base[right], fitted))
            methods[label] = {
                "ratio": ratio_of(transformed, contents[right], layouts[right]),
                "NN_R@1": nearest_neighbour_recall(
                    cosine_distance(transformed), contents[right], layouts[right]
                ),
            }

    return {
        "S_layout": s_layout,
        "S_content": s_content,
        "ratio": ratio,
        "NN_R@1": recall,
        "uniformity_corr": uniformity,
        "per_dim_ratio_spread": spread,
        "per_dim_ratio_median": median_ratio,
        "ratio_after_diagonal": ratio_diag,
        "ratio_after_rotation": ratio_rot,
        "recall_after_rotation": recall_rot,
        "ratio_after_heuristics": heuristics,
        "holdout_methods": methods,
        "holdout_baseline": {
            "ratio": ratio_of(base[right], contents[right], layouts[right]),
            "NN_R@1": nearest_neighbour_recall(
                cosine_distance(base[right]), contents[right], layouts[right]
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Encoder survey on one grid.")
    parser.add_argument("--grid", default="dataset/grid")
    parser.add_argument(
        "--encoders",
        default="clip_openai,dinov2_l",
        help=f"comma list from {sorted(OPEN_CLIP_MODELS) + sorted(DINO_MODELS)}",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0,
                        help="subsample the grid for a quick look (0 = all)")
    parser.add_argument("--report-json", default="outputs/backbone_survey.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths, contents, layouts = load_any_grid(args.grid)
    if args.max_images and len(paths) > args.max_images:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(paths), args.max_images, replace=False))
        paths = [paths[i] for i in keep]
        contents = np.asarray(contents)[keep].tolist()
        layouts = np.asarray(layouts)[keep].tolist()
    print(f"{len(paths)} images, {len(set(contents))} contents, "
          f"{len(set(layouts))} layouts")

    results = {}
    failures = {}
    for name in [item.strip() for item in args.encoders.split(",") if item.strip()]:
        try:
            if name in OPEN_CLIP_MODELS:
                model_name, tag, _ = OPEN_CLIP_MODELS[name]
                model, transform = load_open_clip(model_name, tag, device)
                encoder = ("openclip", (model, transform))
            elif name in DINO_MODELS:
                encoder = ("dino", (load_dino(DINO_MODELS[name], device), dino_transform()))
            else:
                print(f"  [skip] unknown encoder {name}")
                continue
            features = encode(paths, encoder, device, args.batch_size)
            results[name] = analyse(features, contents, layouts)
            row = results[name]
            heur = row["ratio_after_heuristics"]
            def pct(value):
                return 100.0 * (value - row["ratio"]) / row["ratio"]
            print(
                f"{name:14s} ratio={row['ratio']:.3f}  NN_R@1={row['NN_R@1']:.3f}  "
                f"spread={row['per_dim_ratio_spread']:.3f}  "
                f"idf={pct(heur['idf']):+.1f}% mask={pct(heur['mask']):+.1f}% "
                f"den={pct(heur['denoise']):+.1f}%  "
                f"oracleDiag={pct(row['ratio_after_diagonal']):+.1f}%  "
                f"rot={ {k: round(v, 3) for k, v in row['ratio_after_rotation'].items()} }"
            )
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"  [fail] {name}: {failures[name]}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({**results, "_failures": failures}, handle,
                  ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")
    if failures:
        print(f"warning: {len(failures)} encoder(s) failed and are missing from "
              f"the report: {sorted(failures)}")
    print()
    print("Held-out comparison (fit on half the contents, score the other half)")
    print(f"  {'encoder':14s} {'method':16s} {'ratio':>7} {'NN_R@1':>8}")
    for name in sorted(results):
        row = results[name]
        entries = [("raw", row["holdout_baseline"])] + [
            (key, value) for key, value in row["holdout_methods"].items()
        ]
        entries.append(
            ("projection r=4",
             {"ratio": row["ratio_after_rotation"].get(4, float("nan")),
              "NN_R@1": row["recall_after_rotation"].get(4, float("nan"))})
        )
        for method, value in entries:
            print(f"  {name:14s} {method:16s} {value['ratio']:>7.3f} "
                  f"{value['NN_R@1']:>8.3f}")


if __name__ == "__main__":
    main()
