import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from backbone_survey import OPEN_CLIP_MODELS, load_open_clip
from topic_layout_probe import load_corpus, render_grid


@torch.no_grad()
def encode_clip(paths, model, transform, device, batch_size):
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc="clip", leave=False):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        tensor = torch.stack([transform(image) for image in images]).to(device)
        vector = model.encode_image(tensor).float()
        chunks.append((vector / vector.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Image-level language-pair check.")
    parser.add_argument("--languages", default="en,zh,fr")
    parser.add_argument("--topics", type=int, default=40)
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--text-length", type=int, default=420)
    parser.add_argument("--encoders", default="clip_openai,siglip")
    parser.add_argument("--render-dir", default="dataset/flores_grid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/language_image_check.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    corpus = load_corpus(
        "flores", languages=languages, length=args.text_length,
        max_topics=args.topics,
    )
    manifest_path = os.path.join(args.render_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = render_grid(
            corpus, len(languages), args.layouts, args.render_dir, seed=0
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    paths = [row["path"] for row in manifest]
    topics = [row["document"].split("::")[0] for row in manifest]
    doc_lang = [languages[int(row["document"].split("::")[1])] for row in manifest]
    layouts = [row["layout"] for row in manifest]
    unique_topics = sorted(set(topics))
    print(f"{len(paths)} images, {len(unique_topics)} topics, languages {languages}")

    index = {}
    for row, (topic, lang, layout) in enumerate(zip(topics, doc_lang, layouts)):
        index[(topic, lang, layout)] = row

    results = {}
    for name in [item.strip() for item in args.encoders.split(",") if item.strip()]:
        try:
            model_name, tag, _ = OPEN_CLIP_MODELS[name]
            model, transform = load_open_clip(model_name, tag, device)
        except Exception as exc:
            print(f"  [fail] {name}: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        features = encode_clip(paths, model, transform, device, args.batch_size)

        table = {}
        for gallery_lang in languages:
            gallery_rows = [
                index[(topic, gallery_lang, 0)]
                for topic in unique_topics
                if (topic, gallery_lang, 0) in index
            ]
            if not gallery_rows:
                continue
            gallery_features = features[gallery_rows]
            gallery_topics = [topic for topic in unique_topics
                              if (topic, gallery_lang, 0) in index]
            for query_lang in languages:
                if query_lang == gallery_lang:
                    continue
                hits = total = 0
                for topic in unique_topics:
                    for layout in range(1, args.layouts):
                        key = (topic, query_lang, layout)
                        if key not in index:
                            continue
                        scores = features[index[key]] @ gallery_features.T
                        hits += int(gallery_topics[int(scores.argmax())] == topic)
                        total += 1
                table[f"{query_lang}->{gallery_lang}"] = (
                    hits / total if total else float("nan")
                )
        results[name] = table
        print(f"  {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(table.items())))

    chance = 1.0 / len(unique_topics)
    print()
    print("=" * 72)
    print(f"Image-level same-topic R@1 by language pair (chance {chance:.3f})")
    print("=" * 72)
    for name, table in results.items():
        print(f"  {name}")
        for pair, value in sorted(table.items()):
            same_script = set(pair.split("->")) <= {"en", "fr"}
            tag = "same script" if same_script else "different script"
            print(f"    {pair:10s} R@1 = {value:.3f}   ({tag})")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"chance": chance, "results": results}, handle, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
