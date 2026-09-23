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

from factor_eval import cosine_distance, load_clip
from topic_layout_probe import load_corpus


def main():
    parser = argparse.ArgumentParser(description="Cross-script control for the "
                                                 "multilingual retrieval result.")
    parser.add_argument("--languages", default="en,zh,fr")
    parser.add_argument("--topics", type=int, default=40)
    parser.add_argument("--text-length", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--clip-weights", default="clip_weights/open_clip_pytorch_model.bin"
    )
    parser.add_argument("--report-json", default="outputs/language_confound.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    corpus = load_corpus(
        "flores", languages=languages, length=args.text_length,
        max_topics=args.topics,
    )
    topics = sorted(corpus)
    print(f"{len(topics)} topics x {len(languages)} languages: {languages}")


    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=args.clip_weights, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    features = {}
    with torch.no_grad():
        for lang_index, lang in enumerate(languages):
            texts = [corpus[topic][lang_index] for topic in topics]
            chunks = []
            for start in range(0, len(texts), args.batch_size):
                tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
                vec = model.encode_text(tokens).float()
                chunks.append(vec / vec.norm(dim=-1, keepdim=True))
            features[lang] = torch.cat(chunks, dim=0)

    def retrieve(query_lang, gallery_lang):
        similarity = features[query_lang] @ features[gallery_lang].T
        winners = similarity.argmax(dim=1).cpu().numpy()
        return float((winners == np.arange(len(topics))).mean())

    results = {}
    for query in languages:
        results[query] = {
            f"-> {gallery}": retrieve(query, gallery)
            for gallery in languages
            if gallery != query
        }


    mixed = {}
    for query in languages:
        others = [lang for lang in languages if lang != query]
        if not others:
            continue
        gallery = torch.cat([features[lang] for lang in others], dim=0)
        similarity = features[query] @ gallery.T
        winners = similarity.argmax(dim=1).cpu().numpy()
        owners = np.concatenate(
            [np.full(len(topics), lang) for lang in others]
        )
        if len(others) == 1:
            accuracy = float((winners == np.arange(len(topics))).mean())
            mixed[query] = {"accuracy": accuracy, "from": others[0]}
        else:

            truth = np.tile(np.arange(len(topics)), len(others))
            is_correct = truth[winners] == np.arange(len(topics))
            owner = owners[winners]
            mixed[query] = {
                "accuracy": float(is_correct.mean()),
                "winner_language_share": {
                    lang: float((owner[is_correct] == lang).mean())
                    if is_correct.any()
                    else 0.0
                    for lang in others
                },
            }

    print()
    print("=" * 70)
    print("Text-only cross-lingual retrieval (chance = 1/%d = %.3f)"
          % (len(topics), 1.0 / len(topics)))
    print("=" * 70)
    for query in languages:
        for key, value in results[query].items():
            print(f"  query {query:3s} {key:14s} R@1 = {value:.3f}")
    print()
    print("  mixed gallery:")
    for query, value in mixed.items():
        extra = value.get("winner_language_share")
        suffix = (
            "  winner languages: "
            + ", ".join(f"{k} {v:.2f}" for k, v in extra.items())
            if extra
            else ""
        )
        print(f"    query {query:3s} R@1 = {value['accuracy']:.3f}{suffix}")

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump({"pairwise": results, "mixed": mixed}, handle, indent=2)
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
