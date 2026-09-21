import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiment_utils import DOMAINS, fit_basis, half_of, load, metrics, project


def classification(matrix, contents, layouts, max_classes=120):


    if len(set(contents.tolist())) > max_classes:
        unique = sorted(set(contents.tolist()))
        rng = np.random.default_rng(0)
        keep = set(rng.permutation(unique)[:max_classes].tolist())
        mask = np.array([c in keep for c in contents])
        matrix, contents, layouts = matrix[mask], contents[mask], np.asarray(layouts)[mask]
    realisations = sorted(set(np.asarray(layouts).tolist()))
    train_layouts = set(realisations[::2])
    mask = np.array([l in train_layouts for l in layouts])
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(matrix[mask].numpy(), contents[mask])
    return float(clf.score(matrix[~mask].numpy(), contents[~mask]))


def clustering(matrix, contents):
    unique = sorted(set(contents.tolist()))
    if len(unique) > 60:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=len(unique), n_init=3, batch_size=512,
                                 random_state=0)
    else:
        kmeans = KMeans(n_clusters=len(unique), n_init=10, random_state=0)
    labels = kmeans.fit_predict(matrix.numpy())
    return {
        "nmi": float(normalized_mutual_info_score(contents, labels)),
        "ari": float(adjusted_rand_score(contents, labels)),
    }


def describe(name, matrix, contents, layouts, max_classes=120):
    row = {
        "classification": classification(matrix, contents, layouts, max_classes),
        "clustering": clustering(matrix, contents),
    }
    row.update(metrics(matrix, contents, layouts))
    print(f"  {name:34s} acc {row['classification']:.3f}  "
          f"NMI {row['clustering']['nmi']:.3f}  ARI {row['clustering']['ari']:.3f}  "
          f"R@1 {row['NN_R@1']:.3f}  ratio {row['ratio']:.2f}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Downstream tasks.")
    parser.add_argument("--encoders", default="clip_openai,siglip,clip_datacomp")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--domains",
                        default="documents,shapes,real scans,large scans,arxiv,"
                                "identity documents")
    parser.add_argument("--max-classes", type=int, default=60,
                        help="cap on the number of content classes for the probe")
    parser.add_argument("--report-json", default="outputs/downstream_tasks.json")
    args = parser.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    wanted = [d.strip() for d in args.domains.replace(";", ",").split(",") if d.strip()]
    results = {}
    for encoder in encoders:
        results[encoder] = {}
        for domain, _, _ in DOMAINS:
            if domain not in wanted:
                continue
            features, contents, layouts = load(domain, encoder)
            basis = fit_basis(features, contents, layouts, rank=args.rank)
            print(f"{encoder} / {domain}")
            results[encoder][domain] = {
                "raw": describe("raw", features, contents, layouts, args.max_classes),
                "rotation": describe("rotation r=4",
                                     project(features, basis, args.rank),
                                     contents, layouts, args.max_classes),
            }

        pooled = []
        for domain in ("real scans", "arxiv", "identity documents"):
            features, contents, layouts = load(domain, encoder)
            pooled.append((features, np.array([f"{domain}::{c}" for c in contents]),
                           np.asarray(layouts)))
        features = torch.cat([p[0] for p in pooled])
        contents = np.concatenate([p[1] for p in pooled])
        layouts = np.concatenate([p[2] for p in pooled])
        basis = fit_basis(features, contents, layouts, rank=args.rank)
        print(f"{encoder} / pooled real contents ({len(set(contents.tolist()))})")
        results[encoder]["pooled"] = {
            "contents": len(set(contents.tolist())),
            "raw": describe("raw", features, contents, layouts, args.max_classes),
            "rotation": describe("rotation r=4", project(features, basis, args.rank),
                                 contents, layouts, args.max_classes),
        }

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
