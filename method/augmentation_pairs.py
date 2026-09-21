import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from backbone_survey import load_any_grid
from experiment_utils import DOMAINS, load, metrics, project
from factor_eval import split_by_content


def augment(image, rng):

    width, height = image.size
    scale = float(rng.uniform(0.7, 1.0))
    new = (max(8, int(width * scale)), max(8, int(height * scale)))
    resized = image.resize(new, Image.BICUBIC)
    left = int(rng.integers(0, max(1, new[0] - 8)))
    top = int(rng.integers(0, max(1, new[1] - 8)))
    cropped = resized.crop((left, top, left + new[0] - 8, top + new[1] - 8))
    array = np.asarray(cropped.convert("RGB")).astype(np.float32)
    array *= float(rng.uniform(0.85, 1.15))
    array += float(rng.uniform(-12, 12))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def encode_augmented(paths, encoder, device, seed=0):
    import open_clip

    from backbone_survey import OPEN_CLIP_MODELS

    model_name, tag, _ = OPEN_CLIP_MODELS[encoder]
    pretrained = "openai" if encoder == "clip_openai" else tag
    model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    rng = np.random.default_rng(seed)
    first, second = [], []
    with torch.no_grad():
        for start in range(0, len(paths), 16):
            batch = paths[start:start + 16]
            images = [Image.open(path).convert("RGB") for path in batch]
            for image in images:
                pair = [augment(image, rng), augment(image, rng)]
                first.append(transform(pair[0]))
                second.append(transform(pair[1]))
            if start % 160 == 0:
                print(f"    augmented {start + len(batch)}/{len(paths)}")
        a = model.encode_image(torch.stack(first).to(device)).float().cpu()
        b = model.encode_image(torch.stack(second).to(device)).float().cpu()
    return a, b


def main():
    parser = argparse.ArgumentParser(description="Augmentation pairs.")
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--domains", default="documents,real captures")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/augmentation_pairs.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    wanted = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    results = {}
    for domain, grid, cache_name in DOMAINS:
        if domain not in wanted:
            continue
        paths, _, _ = load_any_grid(grid)
        for encoder in encoders:
            features, contents, layouts = load(domain, encoder)
            left, _ = split_by_content(contents)
            subset = np.where(left)[0]
            print(f"{domain} / {encoder}: augmenting {len(subset)} images")
            a, b = encode_augmented([paths[i] for i in subset], encoder, device)
            differences = (a - b).float()
            torch.manual_seed(0)
            basis = torch.pca_lowrank(differences, q=args.rank, center=False)[2]
            row = {
                "raw": metrics(features, contents, layouts),
                "augmentation_pairs": metrics(project(features, basis, args.rank),
                                              contents, layouts),
            }
            results[f"{domain}/{encoder}"] = row
            print(f"  raw R@1 {row['raw']['NN_R@1']:.3f} ratio {row['raw']['ratio']:.2f}"
                  f"  -> augmentation pairs R@1 "
                  f"{row['augmentation_pairs']['NN_R@1']:.3f} ratio "
                  f"{row['augmentation_pairs']['ratio']:.2f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
