import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import glob
import json
import random
import re

import numpy as np
import torch

from dataset_generator import (
    build_layout_grid,
    clean_and_check_text,
    create_text_image,
    get_few_valid_fonts,
)
from factor_eval import (
    cosine_distance,
    encode_all,
    load_clip,
    load_models,
)


HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")


def wikipedia_leads(subjects, languages, length=400, timeout=25):


    import urllib.parse
    import urllib.request
    import time

    state = {"last": 0.0}

    def api(lang, params):
        params = dict(params, format="json", action="query")
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        headers = {
            "User-Agent": (
                "layout-probe/0.1 (research script; contact: local user) "
                "python-urllib"
            ),
            "Accept": "application/json",
        }
        last_error = None
        for attempt in range(4):

            pause = 1.2 - (time.time() - state["last"])
            if pause > 0:
                time.sleep(pause)
            state["last"] = time.time()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except Exception as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        raise last_error

    def truncate(text):
        text = " ".join(text.split())
        if len(text) <= length:
            return text
        cut = text[:length]
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        return cut


    titles = {lang: {} for lang in languages}
    if "en" in languages:
        for subject in subjects:
            titles["en"][subject] = subject
    for lang in languages:
        if lang == "en":
            continue
        links = api("en", {"titles": "|".join(subjects), "prop": "langlinks",
                           "lllang": lang, "redirects": 1, "lllimit": "max"})
        for page in links["query"]["pages"].values():
            subject = page.get("title", "")
            entries = page.get("langlinks") or []
            if entries:
                titles[lang][subject] = entries[0]["*"]

    extracts = {}
    for lang in languages:
        wanted = {s: t for s, t in titles[lang].items()}
        if not wanted:
            continue
        data = api(lang, {"titles": "|".join(wanted.values()), "prop": "extracts",
                          "exintro": 1, "explaintext": 1, "redirects": 1,
                          "exlimit": "max"})

        reverse = {t: s for s, t in wanted.items()}
        for page in data["query"]["pages"].values():
            title = page.get("title", "")
            subject = reverse.get(title) or reverse.get(title.replace(" ", "_"))
            if subject is None:
                continue
            text = page.get("extract") or ""
            if text:
                extracts[(subject, lang)] = truncate(text)

    corpus = {}
    for subject in subjects:
        per_language = [
            extracts[(subject, lang)]
            for lang in languages
            if (subject, lang) in extracts and len(extracts[(subject, lang)]) >= length
        ]
        if len(per_language) == len(languages):
            corpus[subject] = per_language
    print(f"wiki corpus: {len(corpus)} subjects x {len(languages)} languages")
    return corpus


WIKI_SUBJECTS = [
    "Photosynthesis", "Association football", "Quantum mechanics",
    "Coffee", "Bicycle", "Antibiotic", "Volcano", "Jazz",
]


FLORES_LANG = {
    "en": "eng_Latn", "zh": "zho_Hans", "fr": "fra_Latn",
    "de": "deu_Latn", "es": "spa_Latn", "ja": "jpn_Jpan",
    "ar": "arb_Arab", "ru": "rus_Cyrl",
}


def load_flores(root="flores200_dataset", languages=("en", "zh", "fr"),
                target_chars=420, max_topics=40, split="devtest"):


    import csv
    import io

    metadata = os.path.join(root, f"metadata_{split}.tsv")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"missing {metadata}")
    with io.open(metadata, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    sentences = {}
    for lang in languages:
        code = FLORES_LANG.get(lang)
        if code is None:
            raise ValueError(f"no FLORES code for {lang}")
        path = os.path.join(root, split, f"{code}.{split}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}")
        with io.open(path, encoding="utf-8") as handle:
            sentences[lang] = [line.strip() for line in handle.read().splitlines()]
        if len(sentences[lang]) != len(rows):
            raise RuntimeError(
                f"{path} has {len(sentences[lang])} lines but the metadata has "
                f"{len(rows)}; they must be aligned"
            )

    by_topic = {}
    for index, row in enumerate(rows):
        topic = (row.get("topic") or "").strip().lower()
        if topic:
            by_topic.setdefault(topic, []).append(index)

    corpus = {}
    for topic in sorted(by_topic, key=lambda name: -len(by_topic[name])):
        chosen = []
        reached = 0
        for index in by_topic[topic]:
            chosen.append(index)
            reached += len(sentences[languages[0]][index])
            if reached >= target_chars:
                break
        if len(chosen) < 2:
            continue
        texts = []
        for lang in languages:
            joined = " ".join(sentences[lang][i] for i in chosen)
            texts.append(joined)
        if min(len(text) for text in texts) < 120:
            continue
        corpus[topic] = texts
        if len(corpus) >= max_topics:
            break
    print(
        f"flores {split}: {len(corpus)} topics x {len(languages)} languages "
        f"(median en length "
        f"{int(np.median([len(v[0]) for v in corpus.values()])) if corpus else 0})"
    )
    return corpus


def read_arrow(path):

    import pyarrow as pa

    try:
        return pa.ipc.open_file(path).read_all()
    except Exception:
        with pa.memory_map(path, "rb") as handle:
            return pa.ipc.open_stream(handle).read_all()


def _renders(font, probe):

    from PIL import Image, ImageDraw

    def draw(character):
        canvas = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(canvas).text((6, 6), character, font=font, fill=255)
        return canvas.tobytes()

    missing = draw("\ue000")
    return draw(probe) != missing


def fonts_for_texts(texts, required_count=20, font_dirs=None):


    from PIL import ImageFont

    latin_only = all(ord(c) < 128 for text in texts for c in text)
    if latin_only:
        return get_few_valid_fonts(required_count=required_count)

    probes = sorted({c for text in texts for c in text if ord(c) > 127})
    probes = probes[:12]

    probes = probes + ["A", "a", "\u00e9"]

    dirs = font_dirs or [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        os.path.expanduser("~/.fonts"),
    ]
    found = []
    for folder in dirs:
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            for name in sorted(files):
                if not name.lower().endswith((".ttf", ".ttc", ".otf")):
                    continue
                path = os.path.join(root, name)
                try:
                    font = ImageFont.truetype(path, 32)
                except Exception:
                    continue
                if all(_renders(font, probe) for probe in probes):
                    found.append(path)
                    if len(found) >= required_count:
                        break
            if len(found) >= required_count:
                break
        if len(found) >= required_count:
            break
    print(f"font pool: {len(found)} fonts covering {len(probes)} sample glyphs")
    return found


def load_corpus(name, max_chars=600, min_chars=200, seed=0, languages=None,
                subjects=None, length=400, corpus_json=None, flores_root=None,
                max_topics=None):


    path = corpus_json or f"topic_corpus_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            corpus = json.load(handle)
        print(f"corpus {name}: {len(corpus)} topics from {path}")
        return corpus

    if name == "wiki":
        cache = os.path.join("dataset", "wiki_corpus.json")
        langs = languages or ["en", "de", "fr", "es"]
        if os.path.exists(cache):
            with open(cache, "r", encoding="utf-8") as handle:
                corpus = json.load(handle)
            print(f"loaded {len(corpus)} wiki subjects from {cache}")
        else:
            corpus = wikipedia_leads(
                subjects or WIKI_SUBJECTS, langs, length=length
            )
            if corpus:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                with open(cache, "w", encoding="utf-8") as handle:
                    json.dump(corpus, handle, ensure_ascii=False, indent=2)
                print(f"cached wiki corpus to {cache}")
        if not corpus:
            raise RuntimeError(
                "no multilingual texts could be fetched (Wikipedia rate limits "
                "aggressively; wait a few minutes and rerun, or drop a "
                "dataset/wiki_corpus.json in place)"
            )
        return corpus

    if name == "flores":
        return load_flores(
            root=flores_root or "flores200_dataset",
            languages=tuple(languages or ["en", "zh", "fr"]),
            target_chars=length,
            max_topics=max_topics or 40,
        )

    if name == "dbpedia":
        pattern = os.path.join(HF_CACHE, "dbpedia_14", "**", "dbpedia_14-train.arrow")
        text_key, label_key = "content", "label"
    elif name == "20ng":
        pattern = os.path.join(
            HF_CACHE, "SetFit___20_newsgroups", "**", "20_newsgroups-train.arrow"
        )
        text_key, label_key = "text", "label_text"
    else:
        raise ValueError(f"unknown corpus {name}")

    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"no cached arrow file for {name} under {HF_CACHE}; "
            "download it once with `datasets.load_dataset(...)` on a machine "
            "that has network, then copy the cache"
        )
    table = read_arrow(candidates[0])

    rng = random.Random(seed)
    topics = {}
    for row in table.to_pylist():
        text = clean_and_check_text(row[text_key])
        if not text or not (min_chars <= len(text) <= max_chars):
            continue
        label = str(row[label_key])
        bucket = topics.setdefault(label, [])
        if len(bucket) >= 40:
            continue
        bucket.append(text)

    for label in topics:
        rng.shuffle(topics[label])
    topics = {k: v for k, v in topics.items() if len(v) >= 5}
    print(f"corpus {name}: {len(topics)} topics with >=5 usable documents")
    return topics


def render_grid(topics, docs_per_topic, layouts, out_dir, seed=0, canvas_height=0):

    fonts = fonts_for_texts(
        [text for texts in topics.values() for text in texts]
    )
    if not fonts:
        raise RuntimeError("no font can render the corpus")
    layout_pool = build_layout_grid(fonts, layouts, seed)

    labels = sorted(topics)[: len(topics)]
    manifest = []
    for topic in labels:
        for doc_index, text in enumerate(topics[topic][:docs_per_topic]):
            doc_id = f"{topic}::{doc_index}"
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)
            folder = os.path.join(out_dir, safe_id)
            os.makedirs(folder, exist_ok=True)
            for layout_id, layout in enumerate(layout_pool):
                image = create_text_image(
                    text=text, canvas_height=canvas_height or None, **layout
                )
                if image is None:
                    continue
                path = os.path.join(folder, f"l{layout_id:02d}.png")
                image.save(path)
                manifest.append(
                    {
                        "topic": topic,
                        "document": doc_id,
                        "layout": layout_id,
                        "path": path,
                    }
                )
    print(f"rendered {len(manifest)} images "
          f"({len(labels)} topics x {docs_per_topic} docs x {layouts} layouts)")
    return manifest


def cell_means(distance, topics, documents, layouts):

    topics = np.asarray(topics)
    documents = np.asarray(documents)
    layouts = np.asarray(layouts)
    same_doc = documents[:, None] == documents[None, :]
    same_topic = topics[:, None] == topics[None, :]
    same_layout = layouts[:, None] == layouts[None, :]
    values = distance.numpy()
    upper = np.triu(np.ones_like(values, dtype=bool), k=1)

    def mean_of(mask):
        selected = values[mask & upper]
        return float(selected.mean()) if selected.size else float("nan")

    return {
        "same_doc_diff_layout": mean_of(same_doc & ~same_layout),
        "same_topic_diff_doc_same_layout": mean_of(
            ~same_doc & same_topic & same_layout
        ),
        "same_topic_diff_doc_diff_layout": mean_of(
            ~same_doc & same_topic & ~same_layout
        ),
        "diff_topic_same_layout": mean_of(~same_topic & same_layout),
        "diff_topic_diff_layout": mean_of(~same_topic & ~same_layout),
    }


def topic_retrieval(distance, topics, documents, layouts, gallery_layout=0):


    topics = np.asarray(topics)
    documents = np.asarray(documents)
    layouts = np.asarray(layouts)
    unique_topics = sorted(set(topics.tolist()))
    held_out = {}
    for topic in unique_topics:
        members = sorted({d for d, t in zip(documents, topics) if t == topic})
        held_out[topic] = members[-1]
    gallery = np.where(
        np.array([d not in held_out.values() for d in documents])
        & (layouts == gallery_layout)
    )[0]
    if len(gallery) == 0:
        return {}

    hits = 0
    total = 0
    for topic in unique_topics:
        doc = held_out[topic]
        for index in np.where(documents == doc)[0]:
            if layouts[index] == gallery_layout:
                continue
            row = distance[index, gallery]
            best = gallery[int(row.argmin())]
            hits += int(topics[best] == topic)
            total += 1
    return {"topic_recall": hits / total if total else float("nan"), "queries": total}


def main():
    parser = argparse.ArgumentParser(description="Topic vs layout probe.")
    parser.add_argument("--corpus", default="dbpedia",
                        choices=["dbpedia", "20ng", "wiki", "flores"])
    parser.add_argument("--flores-root", default="flores200_dataset")
    parser.add_argument(
        "--corpus-json",
        default=None,
        help=(
            "Path to a {topic: [text, ...]} JSON. Defaults to "
            "topic_corpus_<corpus>.json next to this script, which is the only "
            "path that works on a machine without network."
        ),
    )
    parser.add_argument("--languages", default="en,de,fr,es",
                        help="only used with --corpus wiki")
    parser.add_argument("--text-length", type=int, default=400,
                        help="truncate every wiki lead to this many characters")
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--docs-per-topic", type=int, default=4)
    parser.add_argument("--layouts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-height", type=int, default=0)
    parser.add_argument("--render-dir", default="dataset/topic_layout")
    parser.add_argument("--manifest", default=None, help="reuse an existing render")
    parser.add_argument("--checkpoint-dir", default="model")
    parser.add_argument(
        "--clip-weights", default="clip_weights/open_clip_pytorch_model.bin"
    )
    parser.add_argument("--sae-keys", default="logits_clip_img,sparse_codes_clip_img")
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-json", default="outputs/topic_layout_probe.json")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    manifest_path = args.manifest or os.path.join(args.render_dir, "manifest.json")
    if args.manifest and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(f"reusing {len(manifest)} rendered images from {manifest_path}")
    else:
        corpus = load_corpus(
            args.corpus,
            seed=args.seed,
            languages=[item.strip() for item in args.languages.split(",") if item.strip()],
            length=args.text_length,
            corpus_json=args.corpus_json,
            flores_root=args.flores_root,
            max_topics=args.topics,
        )
        if len(corpus) < args.topics:
            raise RuntimeError(f"only {len(corpus)} topics available")
        keep = dict(list(sorted(corpus.items()))[: args.topics])


        if args.corpus in ("wiki", "flores"):
            docs = len(next(iter(keep.values())))
        else:
            docs = args.docs_per_topic
        manifest = render_grid(
            keep,
            docs,
            args.layouts,
            args.render_dir,
            seed=args.seed,
            canvas_height=args.canvas_height,
        )
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    paths = [row["path"] for row in manifest]
    topics = [row["topic"] for row in manifest]
    documents = [row["document"] for row in manifest]
    layouts = [row["layout"] for row in manifest]

    if args.clip_only:
        clip_m, clip_t = load_clip(args.clip_weights, device)
        models = (lambda x: torch.zeros(x.shape[0], 8), clip_t, clip_m, clip_t, None, 0)
        sae_keys = []
    else:
        models = load_models(args.checkpoint_dir, args.clip_weights, device)
        sae_keys = [key.strip() for key in args.sae_keys.split(",") if key.strip()]

    features = encode_all(paths, sae_keys, models, device, args.batch_size)

    tables = {}
    retrieval = {}
    for name in sorted(features):

        distance = cosine_distance(features[name].float())
        tables[name] = cell_means(distance, topics, documents, layouts)
        retrieval[name] = topic_retrieval(distance, topics, documents, layouts)

    print()
    print("=" * 88)
    print("Mean cosine distance between documents")
    print("=" * 88)
    print("  The question: is 'same topic, different document, different layout'")
    print("  smaller than 'different topic, same layout'?  If yes, topic beats")
    print("  layout across documents and not just across renderings.")
    print()
    keys = list(next(iter(tables.values())).keys())
    width = max(len(k) for k in keys) + 2
    names = sorted(tables)
    label_width = max(len(n) for n in names) + 2
    print(f"  {'cell':<{width}}" + "".join(f"{n:>{label_width}.{label_width}}" for n in names))
    print("  " + "-" * (width + label_width * len(names)))
    for key in keys:
        row = f"  {key:<{width}}"
        for name in names:
            row += f"{tables[name][key]:>{label_width}.4f}"
        print(row)

    print()
    print("  verdict (topic across documents vs layout across renderings):")
    for name in names:
        t = tables[name]
        topic_side = t["same_topic_diff_doc_diff_layout"]
        layout_side = t["diff_topic_same_layout"]
        wins = topic_side < layout_side
        print(
            f"    {name:<{label_width}} same-topic={topic_side:.4f}  "
            f"same-layout={layout_side:.4f}  -> "
            f"{'TOPIC wins' if wins else 'LAYOUT wins'} "
            f"(margin {layout_side - topic_side:+.4f})"
        )

    print()
    print("=" * 88)
    print("Topic retrieval across documents and layouts")
    print("=" * 88)
    print("  gallery = layout 0 of every non-held-out document;")
    print("  query   = another rendering of a held-out document;")
    print("  correct = the top-1 gallery item shares the query's topic.")
    print()
    for name in names:
        r = retrieval[name]
        print(
            f"    {name:<{label_width}} topic_recall={r['topic_recall']:.4f}  "
            f"({r['queries']} queries, {len(set(topics))} topics)"
        )

    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(
            {"tables": tables, "retrieval": retrieval, "manifest": manifest_path},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nreport json: {args.report_json}")


if __name__ == "__main__":
    main()
