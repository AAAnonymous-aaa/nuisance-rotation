import os


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import glob
import json
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_clip(clip_weights, device):

    import open_clip

    if not os.path.exists(clip_weights):
        raise FileNotFoundError(
            f"Clip weights not found: {clip_weights}\n"
            "Pass --clip-weights or CLIP_WEIGHTS=/path/to/open_clip_pytorch_model.bin"
        )
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=clip_weights, device=device
    )
    model.eval()
    return model, transform


def _ensure_sparc_importable():


    try:
        import sparc  # noqa: F401
        return
    except ImportError:
        pass
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sparc_repo")
    for extra in (os.path.join(local, "_stubs"), os.path.join(local, "SPARC-main")):
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.append(extra)


def load_grid(root):

    manifest = os.path.join(root, "layout_manifest.json")
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        paths = [record["path"] for record in records]
        contents = [record["sample_id"] for record in records]
        layouts = [record["layout_id"] for record in records]
        print(f"grid manifest: {len(paths)} images, "
              f"{len(set(contents))} contents, {len(set(layouts))} layouts")
        return paths, contents, layouts


    paths, contents, layouts = [], [], []
    for sample_dir in sorted(glob.glob(os.path.join(root, "sample_*"))):
        if not os.path.isdir(sample_dir):
            continue
        content = os.path.basename(sample_dir)
        images = sorted(
            glob.glob(os.path.join(sample_dir, "*.png"))
            + glob.glob(os.path.join(sample_dir, "*.jpg"))
        )
        for index, path in enumerate(images):
            paths.append(path)
            contents.append(content)
            layouts.append(index)
    if not paths:
        raise FileNotFoundError(f"No grid found under {root}")
    print(f"directory convention: {len(paths)} images, "
          f"{len(set(contents))} contents, {len(set(layouts))} layouts")
    print("  [warn] no layout_manifest.json; assuming var_NN means layout NN. "
          "Only correct if the dataset was generated with --layout-grid.")
    return paths, contents, layouts


def load_models(checkpoint_dir, clip_weights, device):
    _ensure_sparc_importable()
    from extract_fingerprints import setup_image_models

    return setup_image_models(
        {
            "checkpoint_dir": checkpoint_dir,
            "clip_weights": clip_weights,
            "device": device,
        }
    )


@torch.no_grad()
def encode_all(paths, sae_keys, models, device, batch_size):
    dino_m, dino_t, clip_m, clip_t, sparc_m, _ = models
    buckets = {}
    missing = set()
    for start in tqdm(range(0, len(paths), batch_size), desc="Encoding"):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        dino_images = torch.stack([dino_t(image) for image in images]).to(device)
        clip_images = torch.stack([clip_t(image) for image in images]).to(device)

        dino_features = dino_m(dino_images).float()
        clip_features = clip_m.encode_image(clip_images).float()
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

        buckets.setdefault("clip", []).append(clip_features.cpu())
        buckets.setdefault("dino", []).append(dino_features.cpu())

        if sparc_m is not None:
            output = sparc_m(
                {
                    "dino": dino_features,
                    "clip_img": clip_features,
                    "clip_txt": torch.zeros(len(batch), 768, device=device),
                }
            )
            for key in sae_keys:
                if key not in output:
                    missing.add(key)
                    continue
                buckets.setdefault(f"sae:{key}", []).append(
                    output[key].float().cpu()
                )

    if missing:
        print(f"  [warn] keys missing from the model output: {sorted(missing)}")
    return {name: torch.cat(chunks, dim=0) for name, chunks in buckets.items()}


def cosine_distance(matrix):

    normed = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-9)
    n = normed.shape[0]
    out = torch.empty(n, n)
    step = 256
    for start in range(0, n, step):
        block = normed[start : start + step] @ normed.T
        out[start : start + step] = 1.0 - block
    return out


def pair_means(distance, contents, layouts):
    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    same_content = contents[:, None] == contents[None, :]
    same_layout = layouts[:, None] == layouts[None, :]
    values = distance.numpy()

    layout_pairs = values[same_content & ~same_layout]
    content_pairs = values[~same_content & same_layout]
    both_pairs = values[~same_content & ~same_layout]
    return (
        float(layout_pairs.mean()),
        float(content_pairs.mean()),
        float(both_pairs.mean()),
    )


def nearest_neighbour_recall(distance, contents, layouts, gallery_layout=0):

    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    gallery = np.where(layouts == gallery_layout)[0]
    gallery_contents = contents[gallery]
    hits = 0
    total = 0
    for index in range(len(contents)):
        if layouts[index] == gallery_layout:
            continue
        row = distance[index, gallery]
        if gallery_contents[int(row.argmin())] == contents[index]:
            hits += 1
        total += 1
    return hits / total if total else float("nan")


def nearest_neighbour_recall_full(distance, contents, layouts):


    contents = np.asarray(contents)
    hits = 0
    total = 0
    for index in range(len(contents)):
        mask = torch.ones(distance.shape[0], dtype=torch.bool)
        mask[index] = False
        row = distance[index][mask]
        if row.numel() == 0:
            continue
        gallery = torch.nonzero(mask).squeeze(1)
        best = int(gallery[int(row.argmin())])
        if contents[best] == contents[index]:
            hits += 1
        total += 1
    return hits / total if total else float("nan")


def idf_weights(matrix):

    n_documents = matrix.shape[0]
    df = (matrix > 0).sum(dim=0).float()
    return torch.log((n_documents + 1.0) / (df + 1.0)) + 1.0


def layout_differences(matrix, contents, layouts, n_pairs=4000, seed=0):


    contents = np.asarray(contents)
    groups = {}
    for index, content in enumerate(contents):
        groups.setdefault(content, []).append(index)
    keys = [key for key, members in groups.items() if len(members) > 1]
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    left, right = [], []
    for _ in range(n_pairs):
        key = keys[int(rng.integers(0, len(keys)))]
        members = groups[key]
        i, j = rng.choice(len(members), size=2, replace=False)
        left.append(members[i])
        right.append(members[j])
    return matrix[left] - matrix[right]


def remove_subspace(matrix, basis):

    if basis is None or basis.shape[1] == 0:
        return matrix
    coefficients = matrix @ basis
    return matrix - coefficients @ basis.T


def _subspace_metrics(base, basis, rank, contents, layouts):

    if base is None:
        return {}
    if rank <= 0 or basis is None:
        reduced = base
    else:
        reduced = remove_subspace(base, basis[:, :rank])
    reduced = reduced / (reduced.norm(dim=-1, keepdim=True) + 1e-9)
    distance = cosine_distance(reduced)
    s_layout, s_content, _ = pair_means(distance, contents, layouts)
    return {
        "ratio": s_content / s_layout if s_layout > 0 else float("nan"),
        "NN_R@1": nearest_neighbour_recall(distance, contents, layouts),
    }


def split_by_content(contents, fraction=0.5, seed=0):

    contents = np.asarray(contents)
    unique = sorted(set(contents.tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    cut = max(1, int(len(unique) * fraction))
    first = {unique[index] for index in perm[:cut]}
    left = np.array([i for i, c in enumerate(contents) if c in first])
    right = np.array([i for i, c in enumerate(contents) if c not in first])
    return left, right


def calibration_curve(features, contents, layouts, ranks=(8, 32), seed=0):


    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    left, right = split_by_content(contents, fraction=0.7, seed=seed)
    pool = sorted(set(contents[left].tolist()))
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(pool))

    baseline = _subspace_metrics(
        features[right], None, 0, contents[right], layouts[right]
    )["NN_R@1"]
    points = []
    for fraction in (0.15, 0.35, 0.6, 1.0):
        count = max(3, int(len(pool) * fraction))
        chosen = {pool[index] for index in order[:count]}
        subset = np.array([i for i, c in enumerate(contents) if c in chosen])
        differences = layout_differences(
            features[subset], contents[subset], layouts[subset], n_pairs=4000
        )
        if differences is None or differences.shape[0] < 2:
            continue
        top = min(max(ranks), differences.shape[1], differences.shape[0] - 1)
        basis = torch.pca_lowrank(differences, q=max(2, top), center=False)[2]
        entry = {"n_calibration": count, "r0": baseline}
        for rank in ranks:
            entry[f"r{rank}"] = _subspace_metrics(
                features[right], basis, rank, contents[right], layouts[right]
            )["NN_R@1"]
        points.append(entry)
    return points


def prototype_recall(matrix, contents, layouts, query_layout=0, k_values=(1, 2, 4, 8)):


    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    unique = sorted(set(contents.tolist()))
    query_index = np.where(layouts == query_layout)[0]
    if len(query_index) == 0:
        return {}
    rng = np.random.default_rng(0)

    pool_size = max(
        1, int(((contents == unique[0]) & (layouts != query_layout)).sum())
    )
    curve = {}
    for k in tuple(k_values) + (pool_size,):
        prototypes = []
        for content in unique:
            pool = np.where((contents == content) & (layouts != query_layout))[0]
            take = pool if k >= len(pool) else rng.choice(pool, size=k, replace=False)
            prototypes.append(matrix[take].mean(dim=0))
        prototypes = torch.stack(prototypes)
        prototypes = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-9)
        winners = (matrix[query_index] @ prototypes.T).argmax(dim=1).numpy()
        truth = np.searchsorted(unique, contents[query_index])
        curve[k] = float((winners == truth).mean())
    return curve


def variance_decomposition(matrix, contents, layouts):


    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    unique_contents = sorted(set(contents.tolist()))
    unique_layouts = sorted(set(layouts.tolist()))

    overall_content_mean = torch.zeros(matrix.shape[1])
    layout_var = torch.zeros(matrix.shape[1])
    for content in unique_contents:
        block = matrix[contents == content]
        mean = block.mean(dim=0)
        overall_content_mean += mean
        layout_var += ((block - mean) ** 2).mean(dim=0)
    layout_var /= len(unique_contents)

    content_var = torch.zeros(matrix.shape[1])
    for layout in unique_layouts:
        block = matrix[layouts == layout]
        mean = block.mean(dim=0)
        content_var += ((block - mean) ** 2).mean(dim=0)
    content_var /= len(unique_layouts)
    return layout_var, content_var


def main():
    parser = argparse.ArgumentParser(
        description="Content vs layout two-factor evaluation."
    )
    parser.add_argument("--dataset-root", default="dataset/grid")
    parser.add_argument("--checkpoint-dir", default="model")
    parser.add_argument(
        "--clip-weights", default="clip_weights/open_clip_pytorch_model.bin"
    )
    parser.add_argument(
        "--sae-keys", default="logits_clip_img,sparse_codes_clip_img"
    )
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    paths, contents, layouts = load_grid(args.dataset_root)
    contents = np.asarray(contents)
    layouts = np.asarray(layouts)
    n_contents = len(set(contents))
    n_layouts = len(set(layouts))

    if args.clip_only:
        clip_m, clip_t = load_clip(args.clip_weights, device)
        models = (lambda x: torch.zeros(x.shape[0], 8), clip_t, clip_m, clip_t, None, 0)
        sae_keys = []
    else:
        models = load_models(args.checkpoint_dir, args.clip_weights, device)
        sae_keys = [key.strip() for key in args.sae_keys.split(",") if key.strip()]

    features = encode_all(paths, sae_keys, models, device, args.batch_size)


    layouts_array = np.asarray(layouts)
    gallery_index = np.where(layouts_array == 0)[0]

    rows = []
    dim_rows = []
    prototype_rows = []
    subspace_rows = []
    subspace_ranks = (0, 4, 8, 16, 32, 64)
    for name in sorted(features):
        base = features[name].float()
        base = base / (base.norm(dim=-1, keepdim=True) + 1e-9)


        layout_var, content_var = variance_decomposition(
            features[name].float(), contents, layouts
        )
        selectivity = content_var / (layout_var + 1e-12)
        selectivity = selectivity / (selectivity.mean() + 1e-12)
        selectivity = selectivity.clamp(0.1, 10.0).sqrt()

        variants = [(name, base)]

        weights = idf_weights(base[gallery_index])
        weighted = base * weights
        weighted = weighted / (weighted.norm(dim=-1, keepdim=True) + 1e-9)
        variants.append((f"{name} + IDF", weighted))

        selective = base * selectivity
        selective = selective / (selective.norm(dim=-1, keepdim=True) + 1e-9)
        variants.append((f"{name} + content-selective", selective))

        for variant_name, matrix in variants:
            distance = cosine_distance(matrix)
            s_layout, s_content, s_both = pair_means(distance, contents, layouts)
            recall = nearest_neighbour_recall(distance, contents, layouts)
            recall_full = nearest_neighbour_recall_full(distance, contents, layouts)
            rows.append(
                {
                    "representation": variant_name,
                    "S_layout": s_layout,
                    "S_content": s_content,
                    "S_both": s_both,
                    "margin": s_content - s_layout,
                    "ratio": s_content / s_layout if s_layout > 0 else float("nan"),
                    "NN_R@1": recall,
                    "NN_R@1_full_gallery": recall_full,
                }
            )
            curve = prototype_recall(matrix, contents, layouts)
            if curve:
                prototype_rows.append(
                    {"representation": variant_name, "curve": curve}
                )

        order = torch.argsort(weights, descending=True)
        quarter = max(1, len(order) // 4)
        kept, suppressed = order[:quarter], order[-quarter:]


        differences = layout_differences(
            features[name].float(), contents, layouts, n_pairs=4000
        )
        basis = None
        if differences is not None and differences.shape[0] > 1:
            max_rank = min(subspace_ranks[-1], differences.shape[1],
                           differences.shape[0] - 1)
            _, _, directions = torch.pca_lowrank(
                differences, q=max(2, max_rank), center=False
            )
            basis = directions

        left, right = split_by_content(contents)
        holdout_basis = None
        raw = features[name].float()
        differences_left = layout_differences(
            raw[left], contents[left], layouts[left], n_pairs=4000
        )
        if differences_left is not None and differences_left.shape[0] > 1:
            left_rank = min(
                subspace_ranks[-1], differences_left.shape[1],
                differences_left.shape[0] - 1
            )
            _, _, directions_left = torch.pca_lowrank(
                differences_left, q=max(2, left_rank), center=False
            )
            holdout_basis = directions_left
        subspace_rows.append(
            {
                "representation": name,
                "calibration": calibration_curve(raw, contents, layouts),
                "points": {
                    rank: _subspace_metrics(base, basis, rank, contents, layouts)
                    for rank in subspace_ranks
                },
                "points_holdout": {
                    rank: _subspace_metrics(
                        base[right],
                        holdout_basis,
                        rank,
                        contents[right],
                        layouts[right],
                    )
                    for rank in subspace_ranks
                }
                if holdout_basis is not None
                else {},
            }
        )

        weight_array = weights.numpy()
        layout_array = layout_var.numpy()
        if weight_array.std() > 0 and layout_array.std() > 0:
            corr = float(np.corrcoef(weight_array, layout_array)[0, 1])
        else:
            corr = float("nan")
        dim_rows.append(
            {
                "representation": name,
                "mean_layout_var": float(layout_var.mean()),
                "mean_content_var": float(content_var.mean()),
                "content_over_layout": float(
                    content_var.mean() / (layout_var.mean() + 1e-12)
                ),
                "layout_var_of_idf_kept": float(layout_var[kept].mean()),
                "layout_var_of_idf_suppressed": float(layout_var[suppressed].mean()),
                "corr_idf_vs_layout_var": corr,
            }
        )

    print()
    print("=" * 78)
    print("TABLE 1  content vs layout geometry   (cosine distance)")
    print("=" * 78)
    print("  S_layout  same content, different layout   -> want SMALL")
    print("  S_content same layout,  different content  -> want LARGE")
    print("  '+ content-selective' reweights each dimension by its measured")
    print("  content/layout ratio on this same grid, i.e. it does on purpose what")
    print("  plan2's denoise was supposed to do. It is supervised by the grid.")
    print("  NN_R@1    query = any rendering, gallery = layout 0 of each content")
    print("            (every candidate is therefore in a DIFFERENT layout than the")
    print("             query, so only content can decide the match)")
    print("  NN_full   same query, gallery = every other rendering, i.e. candidates")
    print("            that may share the query's layout are available again.")
    print("            NN_full >> NN_R@1 means the task is being solved by matching")
    print("            layout, not content.")
    print()
    header = (
        f"  {'representation':<34}{'S_layout':>10}{'S_content':>11}{'S_both':>9}"
        f"{'ratio':>8}{'NN_R@1':>9}{'NN_full':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(
            f"  {row['representation']:<34}{row['S_layout']:>10.4f}"
            f"{row['S_content']:>11.4f}{row['S_both']:>9.4f}"
            f"{row['ratio']:>8.2f}{row['NN_R@1']:>9.4f}"
            f"{row['NN_R@1_full_gallery']:>9.4f}"
        )

    if prototype_rows:
        keys = sorted(prototype_rows[0]["curve"])
        print()
        print("  S_both   different content AND different layout -- the 'everything")
        print("           differs' cell. If it is barely above S_layout and S_content,")
        print("           the representation is not ordering these relationships at all.")
        print()
        print("=" * 78)
        print("TABLE 1b  how many gallery renderings per content does it take?")
        print("=" * 78)
        print("  Each prototype is the mean of k renderings of one content; the query is")
        print("  a held-out rendering in another layout. plan2's database uses k = 9-ish,")
        print("  so this shows how much of its ~90% was the averaging.")
        print()
        print(
            f"  {'representation':<34}"
            + "".join(f"{'k=' + str(k):>8}" for k in keys)
        )
        print("  " + "-" * (34 + 8 * len(keys)))
        for entry in prototype_rows:
            print(
                f"  {entry['representation']:<34}"
                + "".join(f"{entry['curve'][k]:>8.4f}" for k in keys)
            )

    if subspace_rows:
        ranks = list(subspace_rows[0]["points"])
        print()
        print("=" * 78)
        print("TABLE 4  projecting out the layout subspace")
        print("=" * 78)
        print("  The top-r directions of 'same content, different layout' differences")
        print("  are removed before scoring (r=0 is the raw feature). This is the")
        print("  closed-form rotation a trained projection head would have to learn,")
        print("  so if the ratio does not move here, training will not move it either.")
        print("  'holdout' fits the subspace on half the contents and scores the")
        print("  other half; 'fitted' uses all contents and is therefore optimistic.")
        print()
        for metric in ("ratio", "NN_R@1"):
            print(f"  {metric}")
            print(
                f"    {'representation':<34}"
                + "".join(f"{'r=' + str(r):>8}" for r in ranks)
            )
            print("    " + "-" * (32 + 8 * len(ranks)))
            for entry in subspace_rows:
                print(
                    f"    {entry['representation'] + ' fitted':<34}"
                    + "".join(
                        f"{entry['points'][r][metric]:>8.4f}" for r in ranks
                    )
                )
                if entry["points_holdout"]:
                    print(
                        f"    {entry['representation'] + ' holdout':<34}"
                        + "".join(
                            f"{entry['points_holdout'][r][metric]:>8.4f}"
                            for r in ranks
                        )
                    )
            print()

    if subspace_rows and subspace_rows[0]["calibration"]:
        print()
        print("=" * 78)
        print("TABLE 5  how much calibration data does the projection need?")
        print("=" * 78)
        print("  A fixed 30% of the contents is the test set; the layout subspace is")
        print("  refitted from increasing subsets of the remaining 70%. r0 is the")
        print("  uncalibrated baseline on the same test contents.")
        print()
        for entry in subspace_rows:
            points = entry["calibration"]
            columns = [key for key in points[0] if key != "n_calibration"]
            print(f"  {entry['representation']}")
            print(
                f"    {'n_cal':>6}" + "".join(f"{key:>9}" for key in columns)
            )
            for point in points:
                print(
                    f"    {point['n_calibration']:>6}"
                    + "".join(f"{point[key]:>9.4f}" for key in columns)
                )
        print()

    print()
    print("=" * 78)
    print("TABLE 2  per-dimension variance decomposition")
    print("=" * 78)
    print("  layout_var  how much a dimension moves when only the layout changes")
    print("  content_var how much it moves when only the content changes")
    print()
    header = f"  {'representation':<34}{'content/layout':>15}{'corr(idf,layoutvar)':>21}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in dim_rows:
        print(
            f"  {row['representation']:<34}{row['content_over_layout']:>15.2f}"
            f"{row['corr_idf_vs_layout_var']:>21.3f}"
        )

    print()
    print("=" * 78)
    print("TABLE 3  is IDF a layout suppressor?")
    print("=" * 78)
    print("  Dims are split by IDF weight. If IDF suppresses layout, the dims it")
    print("  keeps should have LOWER layout variance than the ones it suppresses.")
    print()
    header = (
        f"  {'representation':<34}{'layout var of':>15}{'layout var of':>15}"
        f"{'verdict':>12}"
    )
    print(f"  {'':<34}{'IDF-kept':>15}{'IDF-suppressed':>15}{'':>12}")
    print("  " + "-" * (len(header) - 2))
    for row in dim_rows:
        kept = row["layout_var_of_idf_kept"]
        suppressed = row["layout_var_of_idf_suppressed"]
        verdict = "yes" if kept < suppressed else "no"
        print(
            f"  {row['representation']:<34}{kept:>15.4g}{suppressed:>15.4g}"
            f"{verdict:>12}"
        )

    report_path = args.report_json or os.path.join("outputs", "factor_eval.json")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": os.path.abspath(args.dataset_root),
                "n_contents": n_contents,
                "n_layouts": n_layouts,
                "geometry": rows,
                "per_dimension": dim_rows,
                "prototype_curve": prototype_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nreport json: {report_path}")


if __name__ == "__main__":
    main()
