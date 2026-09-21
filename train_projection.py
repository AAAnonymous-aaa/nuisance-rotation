import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time

import numpy as np
import torch

from backbone_survey import analyse
from seed_variance import cached_features


class Projection(torch.nn.Module):


    def __init__(self, dim, hidden=0, diagonal=False):
        super().__init__()
        self.diagonal = diagonal
        if diagonal:
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.residual = False
        elif hidden:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, dim),
            )
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)
            self.residual = True
        else:
            self.net = torch.nn.Linear(dim, dim)
            torch.nn.init.eye_(self.net.weight)
            torch.nn.init.zeros_(self.net.bias)
            self.residual = False

    def forward(self, x):
        if self.diagonal:
            return x * self.weight
        out = self.net(x)
        return x + out if self.residual else out


def supervised_contrastive(features, labels, temperature):

    z = torch.nn.functional.normalize(features, dim=-1)
    similarity = (z @ z.t()) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    same = labels[:, None] == labels[None, :]
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=features.device)
    positives = same & ~self_mask
    exp = torch.exp(similarity)
    exp = exp.masked_fill(self_mask, 0.0)
    denominator = exp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = similarity - denominator.log()
    counts = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives).sum(dim=1) / counts
    return loss.mean()


def main():
    parser = argparse.ArgumentParser(description="Train a projection head.")
    parser.add_argument("--train-grid", default="dataset/sae_train")
    parser.add_argument("--eval-grid", default="survey_grid")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--exclude-docs", type=int, default=6,
                        help="skip documents 0..N-1, which the eval grid uses")
    parser.add_argument("--contents-per-batch", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden", type=int, default=0,
                        help="0 for a linear projection, >0 for a 2-layer MLP")
    parser.add_argument("--diagonal", action="store_true",
                        help="train a per-dimension rescaling instead of a mixing head")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the batch sampler (training run)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/trained_projection.pt")
    parser.add_argument("--report-json", default="outputs/trained_projection.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_features, train_contents, train_layouts = cached_features(
        args.train_grid, "sae_train", args.encoder, device, "outputs/cache"
    )
    keep = [
        index for index, document in enumerate(train_contents)
        if int(document.split("::")[1]) >= args.exclude_docs
    ]
    train_features = train_features[keep]
    train_contents = [train_contents[index] for index in keep]
    train_layouts = [train_layouts[index] for index in keep]
    print(f"train on {len(train_features)} images, "
          f"{len(set(train_contents))} contents (documents {args.exclude_docs}+)")

    unique_contents = sorted(set(train_contents))
    by_content = {c: [i for i, d in enumerate(train_contents) if d == c]
                  for c in unique_contents}
    data = train_features.to(device)

    torch.manual_seed(args.seed)
    model = Projection(data.shape[1], args.hidden, args.diagonal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    steps = max(1, len(unique_contents) // args.contents_per_batch)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(unique_contents))
        total = 0.0
        for step in range(steps):
            chunk = order[step * args.contents_per_batch :
                          (step + 1) * args.contents_per_batch]
            indices, labels = [], []
            for position, content_index in enumerate(chunk):
                members = by_content[unique_contents[content_index]]
                chosen = rng.choice(members, size=2, replace=False)
                indices.extend(int(i) for i in chosen)
                labels.extend([position, position])
            batch = data[indices]
            loss = supervised_contrastive(model(batch), torch.tensor(labels, device=device),
                                          args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={total / steps:.4f}  "
                  f"({time.time() - started:.0f}s)")

    peak = (
        torch.cuda.max_memory_allocated() / 1024**2
        if device.type == "cuda" else float("nan")
    )
    reserved = (
        torch.cuda.max_memory_reserved() / 1024**2
        if device.type == "cuda" else float("nan")
    )
    print(f"training finished in {time.time() - started:.0f}s; "
          f"peak GPU memory allocated {peak:.0f} MB, reserved {reserved:.0f} MB")

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.out)


    eval_features, eval_contents, eval_layouts = cached_features(
        args.eval_grid, "documents", args.encoder, device, "outputs/cache"
    )
    with torch.no_grad():
        projected = model(eval_features.to(device)).cpu()
    results = {}
    for name, matrix in (("raw_features", eval_features),
                         ("trained_projection", projected)):
        row = analyse(matrix, eval_contents, eval_layouts)
        results[name] = {
            key: row[key] for key in
            ("ratio", "NN_R@1", "per_dim_ratio_spread",
             "ratio_after_diagonal", "ratio_after_rotation")
        }
        print(f"  {name:20s} ratio={row['ratio']:.3f}  NN_R@1={row['NN_R@1']:.3f}  "
              f"spread={row['per_dim_ratio_spread']:.3f}")
    results["peak_gpu_mb"] = peak
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
