import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from backbone_survey import OPEN_CLIP_MODELS, analyse, load_any_grid, load_open_clip
from topic_layout_probe import load_corpus, render_grid


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_in, n_latents, k, auxk):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.k = k
        self.auxk = auxk
        self.encoder = torch.nn.Linear(d_in, n_latents, bias=True)

        self.pre_bias = torch.nn.Parameter(torch.zeros(d_in))

    def forward(self, x):
        centred = x - self.pre_bias
        logits = self.encoder(centred)
        values, indices = torch.topk(logits, self.k, dim=1)
        sparse = torch.zeros_like(logits)
        sparse.scatter_(1, indices, torch.relu(values))
        recon = sparse @ self.encoder.weight + self.pre_bias

        residual = x - recon
        return logits, sparse, recon, residual


def unit_norm_columns(module):
    with torch.no_grad():
        module.weight.div_(module.weight.norm(dim=1, keepdim=True) + 1e-8)


@torch.no_grad()
def encode_features(paths, model, transform, device, batch_size):
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="features", leave=False):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        tensor = torch.stack([transform(image) for image in images]).to(device)
        vector = model.encode_image(tensor).float()
        chunks.append((vector / vector.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Train a document-domain SAE.")
    parser.add_argument("--grid", default="./survey_grid")
    parser.add_argument("--corpus", default="dbpedia")
    parser.add_argument("--topics", type=int, default=14)
    parser.add_argument("--docs-per-topic", type=int, default=40)
    parser.add_argument("--layouts", type=int, default=6)
    parser.add_argument("--exclude-docs", type=int, default=6,
                        help="documents 0..N-1 are the ones the eval grid uses")
    parser.add_argument("--render-dir", default="dataset/sae_train")
    parser.add_argument("--encoder", default="clip_openai")
    parser.add_argument("--clip-weights", default="clip_weights/open_clip_pytorch_model.bin")
    parser.add_argument("--n-latents", type=int, default=8192)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--auxk", type=int, default=64)
    parser.add_argument("--aux-coef", type=float, default=0.03125)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/domain_sae.pt")
    parser.add_argument("--report-json", default="outputs/domain_sae_survey.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_name, tag, _ = OPEN_CLIP_MODELS[args.encoder]
    clip_model, clip_transform = load_open_clip(
        model_name, "openai" if args.encoder == "clip_openai" else tag, device
    )

    manifest_path = os.path.join(args.render_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        corpus = load_corpus(
            args.corpus, languages=["en"], length=420, max_topics=args.topics
        )
        corpus = {topic: texts for topic, texts in corpus.items()}
        manifest = render_grid(
            corpus, args.docs_per_topic, args.layouts, args.render_dir, seed=99
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    keep = [
        row for row in manifest
        if int(row["document"].split("::")[1]) >= args.exclude_docs
    ]
    print(f"training on {len(keep)} images "
          f"(documents {args.exclude_docs}+ of each topic)")
    train_features = encode_features(
        [row["path"] for row in keep], clip_model, clip_transform, device,
        args.batch_size
    )

    sae = SparseAutoencoder(
        train_features.shape[1], args.n_latents, args.k, args.auxk
    ).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    data = train_features.to(device)
    n = data.shape[0]
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = data[order[start : start + args.batch_size]]
            logits, sparse, recon, residual = sae(batch)
            loss = torch.nn.functional.mse_loss(recon, batch)
            total += float(loss) * batch.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            unit_norm_columns(sae.encoder)
        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                active = (sparse > 0).float().sum(dim=0)
                dead = float((active == 0).float().mean())
            print(f"  epoch {epoch:4d}  mse={total / n:.5f}  dead latents={dead:.3%}"
                  f"  ({time.time() - started:.0f}s)")

    torch.save({"state_dict": sae.state_dict(), "args": vars(args)},
               args.out)
    print(f"saved {args.out}")


    paths, contents, layouts = load_any_grid(args.grid)
    grid_features = encode_features(
        paths, clip_model, clip_transform, device, args.batch_size
    )
    with torch.no_grad():
        logits, sparse, _, _ = sae(grid_features.to(device))
    results = {
        "domain_sae_logits": analyse(logits.cpu(), contents, layouts),
        "domain_sae_sparse": analyse(sparse.cpu(), contents, layouts),
    }
    for name, row in results.items():
        print(f"{name:20s} ratio={row['ratio']:.3f}  NN_R@1={row['NN_R@1']:.3f}  "
              f"spread={row['per_dim_ratio_spread']:.3f}")
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
