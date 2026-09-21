import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import shutil
import time
import urllib.request

import numpy as np
import pypdfium2 as pdfium
from PIL import Image


SETTINGS = [
    {"scale": 1.0, "gray": False, "border": 0},
    {"scale": 1.5, "gray": False, "border": 0},
    {"scale": 1.0, "gray": True, "border": 12},
    {"scale": 1.5, "gray": True, "border": 24},
]


def fetch_pdfs(categories, per_category, pdf_dir, delay=3.0):
    import arxiv

    client = arxiv.Client(page_size=per_category, delay_seconds=delay,
                          num_retries=3)
    records = []
    for category in categories:
        query = f"cat:{category}"
        search = arxiv.Search(query=query, max_results=per_category,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        found = 0
        for result in client.results(search):
            if found >= per_category:
                break
            safe = result.get_short_id().replace("/", "_")
            path = os.path.join(pdf_dir, f"{safe}.pdf")
            if not os.path.exists(path):
                try:
                    url = getattr(result, "pdf_url", None) or (
                        f"https://arxiv.org/pdf/{result.get_short_id()}"
                    )
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "arxiv-scan-domain/0.1"}
                    )
                    with urllib.request.urlopen(request, timeout=90) as response:
                        with open(path, "wb") as handle:
                            shutil.copyfileobj(response, handle)
                    time.sleep(delay)
                except Exception as exc:
                    print(f"  [warn] {safe}: {type(exc).__name__}")
                    continue
            records.append({"category": category, "id": safe, "pdf": path})
            found += 1
            print(f"  [{category}] {found}/{per_category} {safe}")
    return records


def render(pdf_path, setting, out_path):
    document = pdfium.PdfDocument(pdf_path)
    page = document[0]
    image = page.render(scale=setting["scale"]).to_pil()
    document.close()
    if setting["gray"]:
        image = image.convert("L").convert("RGB")
    border = setting["border"]
    if border:
        array = np.asarray(image)
        array = np.pad(array, ((border, border), (border, border), (0, 0)),
                       mode="constant", constant_values=255)
        image = Image.fromarray(array)
    image.save(out_path)
    return image.size


def main():
    parser = argparse.ArgumentParser(description="Real arXiv scan domain.")
    parser.add_argument("--categories", default="cs.CL,math.OC,astro-ph,cond-mat")
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--out", default="dataset/real_scan")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--reuse", action="store_true",
                        help="skip the download and reuse existing pdfs")
    args = parser.parse_args()

    pdf_dir = os.path.join(args.out, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]

    manifest_path = os.path.join(args.out, "manifest.json")
    if args.reuse and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        print(f"reusing {len(manifest)} rendered pages")
    else:
        records = fetch_pdfs(categories, args.per_category, pdf_dir, args.delay)
        manifest = []
        for index, record in enumerate(records):
            folder = os.path.join(args.out, f"p{index:04d}")
            os.makedirs(folder, exist_ok=True)
            for layout_id, setting in enumerate(SETTINGS):
                path = os.path.join(folder, f"l{layout_id:02d}.png")
                if not os.path.exists(path):
                    try:
                        render(record["pdf"], setting, path)
                    except Exception as exc:
                        print(f"  [warn] render {record['id']}: {type(exc).__name__}")
                        continue
                manifest.append(
                    {
                        "topic": record["category"],
                        "document": f"{record['category']}::{index:04d}",
                        "layout": layout_id,
                        "path": path,
                    }
                )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"real scan domain: {len(manifest)} images, "
          f"{len({row['document'] for row in manifest})} papers, "
          f"{len({row['layout'] for row in manifest})} scan settings")


if __name__ == "__main__":
    main()
