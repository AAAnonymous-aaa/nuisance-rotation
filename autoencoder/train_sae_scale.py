import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time

import torch

from experiment_utils import load, metrics
from seed_variance import cached_features
from train_domain_sae import SparseAutoencoder, unit_norm_columns


BUDGETS = (16, 32, 64, 128, 256, 1024, 0)


def sparse_at(logits, k):
    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k, dim=1)
    sparse = torch.zeros_like(logits)
    sparse.scatter_(1, indices, torch.relu(values))
    return sparse


def dead_fraction(sae, features, batch=1024):
    used = torch.zeros(sae.n_latents, dtype=torch.bool)
    with torch.no_grad():
        for start in range(0, len(features), batch):
            logits = sae.encoder(features[start:start + batch] - sae.pre_bias)
            indices = torch.topk(logits, sae.k, dim=1)[1]
            used[indices.reshape(-1).cpu()] = True
    return float(1.0 - used.float().mean())


def main():
    parser = argparse.ArgumentParser(description="Scaled in-domain SAE.")
    parser.add_argument("--grid", default="dataset/sae_large")
    parser.add_argument("--cache-name", default="sae_large")
    parser.add_argument("--eval-domain", default="documents")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--n-latents", type=int, default=8192)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--auxk", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/domain_sae_large.pt")
    parser.add_argument("--report-json", default="outputs/domain_sae_large.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, contents, layouts = cached_features(
        args.grid, args.cache_name, args.encoder, device, "outputs/cache"
    )
    print(f"training SAE on {tuple(features.shape)} from {args.grid}")

    sae = SparseAutoencoder(features.shape[1], args.n_latents, args.k, args.auxk).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    data = features.to(device)
    n = data.shape[0]
    steps_per_epoch = max(1, n // args.batch_size)
    started = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total = 0.0
        for step in range(steps_per_epoch):
            batch = data[order[step * args.batch_size:(step + 1) * args.batch_size]]
            _, _, recon, _ = sae(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            unit_norm_columns(sae.encoder)
            total += float(loss.detach())
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  mse={total / steps_per_epoch:.3e}  "
                  f"({time.time() - started:.0f}s)")
            history.append({"epoch": epoch, "mse": total / steps_per_epoch})

    dead = dead_fraction(sae, data)
    print(f"training done in {time.time() - started:.0f}s; dead latents {dead:.1%}")

    eval_features, eval_contents, eval_layouts = load(args.eval_domain, args.encoder)
    with torch.no_grad():
        logits, sparse, recon, _ = sae(eval_features.to(device))
    logits = logits.cpu()
    eval_mse = float(torch.nn.functional.mse_loss(recon.cpu(), eval_features))

    results = {"config": vars(args), "images": int(n), "dead_latent_fraction": dead,
               "final_train_mse": history[-1]["mse"] if history else None,
               "eval_reconstruction_mse": eval_mse, "history": history, "budgets": {}}
    for k in BUDGETS:
        name = "dense" if k <= 0 else f"k={k}"
        row = metrics(sparse_at(logits, k), eval_contents, eval_layouts)
        results["budgets"][name] = row
        print(f"  {name:6s} ratio={row['ratio']:5.2f}  R@1={row['NN_R@1']:.3f}")
    results["released_sparse_reference"] = float((sparse > 0).sum(dim=1).float().mean())

    torch.save({"state_dict": sae.state_dict(), "args": vars(args)}, args.out)
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nsaved {args.out}\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
