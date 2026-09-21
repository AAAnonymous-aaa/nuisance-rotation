import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from debiasing_baselines import inlp_fit, leace_fit
from experiment_utils import fit_basis, half_of, load, metrics, project
from seed_variance import cached_features
from train_projection import Projection, supervised_contrastive


def probe_accuracy(train_x, train_y, test_x, test_y):

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_x.numpy(), train_y)
    predicted = clf.predict(test_x.numpy())
    classes = np.unique(train_y)
    recalls = [np.mean(predicted[test_y == c] == c) for c in classes
               if (test_y == c).any()]
    return float(np.mean(recalls))


def adversarial_train(features, contents, labels, hidden, epochs, lam, batch_contents,
                      temperature, seed, device):

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    unique = sorted(set(contents))
    by_content = {c: [i for i, d in enumerate(contents) if d == c] for c in unique}
    label_ids = sorted(set(labels))
    label_map = {l: i for i, l in enumerate(label_ids)}
    label_tensor = torch.tensor([label_map[l] for l in labels], device=device)

    head = Projection(features.shape[1], hidden).to(device)
    adversary = torch.nn.Linear(features.shape[1], len(label_ids)).to(device)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=1e-3)
    data = features.to(device)
    steps = max(1, len(unique) // batch_contents)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(unique))
        for step in range(steps):
            chunk = order[step * batch_contents:(step + 1) * batch_contents]
            indices, group = [], []
            for position, content_index in enumerate(chunk):
                members = by_content[unique[content_index]]
                chosen = rng.choice(members, size=2, replace=False)
                indices.extend(int(i) for i in chosen)
                group.extend([position, position])
            batch = data[indices]
            batch_labels = label_tensor[indices]
            group_tensor = torch.tensor(group, device=device)

            with torch.no_grad():
                z = head(batch)
            adversary_loss = torch.nn.functional.cross_entropy(adversary(z), batch_labels)
            adversary_optimizer.zero_grad()
            adversary_loss.backward()
            adversary_optimizer.step()

            z = head(batch)
            content_loss = supervised_contrastive(z, group_tensor, temperature)
            fool_loss = torch.nn.functional.cross_entropy(adversary(z), batch_labels)
            head_loss = content_loss - lam * fool_loss
            head_optimizer.zero_grad()
            head_loss.backward()
            head_optimizer.step()
    return head


def main():
    parser = argparse.ArgumentParser(description="Adversarial head and decodability.")
    parser.add_argument("--train-grid", default="dataset/sae_train")
    parser.add_argument("--train-name", default="sae_train")
    parser.add_argument("--eval-domain", default="documents")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lambda-adversary", type=float, default=1.0)
    parser.add_argument("--contents-per-batch", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/adversarial_head.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_features, train_contents, train_layouts = cached_features(
        args.train_grid, args.train_name, args.encoder, device, "outputs/cache"
    )
    train_contents = np.asarray(train_contents)
    train_layouts = np.asarray(train_layouts)
    keep = np.array([int(d.split("::")[1]) >= 6 for d in train_contents])
    train_features = train_features[keep]
    train_contents = train_contents[keep]
    train_layouts = train_layouts[keep]

    eval_features, eval_contents, eval_layouts = load(args.eval_domain, args.encoder)
    half = half_of(eval_contents)
    print(f"train {tuple(train_features.shape)}, eval {tuple(eval_features.shape)}")

    basis = fit_basis(train_features, train_contents, train_layouts, rank=args.rank)
    representations = {
        "raw": eval_features,
        "closed form r=4": project(eval_features, basis, args.rank),
        "LEACE-style": leace_fit(train_features, train_layouts.tolist())(eval_features),
        "INLP": inlp_fit(train_features, train_layouts.tolist(), rank=20)(eval_features),
    }

    started = time.time()
    head = adversarial_train(
        train_features, train_contents.tolist(), train_layouts.tolist(),
        args.hidden, args.epochs, args.lambda_adversary, args.contents_per_batch,
        args.temperature, args.seed, device,
    )
    print(f"adversarial head trained in {time.time() - started:.0f}s")
    with torch.no_grad():
        representations["adversarial head"] = head(eval_features.to(device)).cpu()

    results = {}
    for name, matrix in representations.items():
        row = metrics(matrix, eval_contents, eval_layouts)
        row["probe_accuracy"] = probe_accuracy(
            matrix[half], eval_layouts[half], matrix[~half], eval_layouts[~half]
        )
        results[name] = row
        print(f"{name:18s} ratio={row['ratio']:5.2f}  R@1={row['NN_R@1']:.3f}  "
              f"nuisance probe acc={row['probe_accuracy']:.3f}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
