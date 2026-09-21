import os
import sys
import argparse
import json
import random
import string
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ALLOWED_CHARS = set(string.ascii_letters + string.digits + string.punctuation + " ")


LAYOUT_VALUES = {
    "font_size": [16, 18, 22, 24, 28],
    "image_width": [600, 750, 800, 900],
    "margin": [20, 30, 45, 60],
    "line_spacing": [1.1, 1.3, 1.5, 1.6],
    "align": ["left", "center", "right"],
}


def random_layout(fonts, rng):
    layout = {"font_path": rng.choice(fonts)}
    for key, values in LAYOUT_VALUES.items():
        layout[key] = rng.choice(values)
    return layout


def layout_key(layout):
    return tuple(sorted(layout.items()))


def build_layout_grid(fonts, count, seed):


    rng = random.Random(0 if seed is None else seed)
    layouts = []
    seen = set()
    attempts = 0
    limit = max(1000, count * 500)
    while len(layouts) < count and attempts < limit:
        attempts += 1
        layout = random_layout(fonts, rng)
        key = layout_key(layout)
        if key in seen:
            continue
        seen.add(key)
        layouts.append(layout)
    if len(layouts) < count:
        raise RuntimeError(
            f"could only draw {len(layouts)} distinct layouts out of {count}; "
            "lower --layout-grid or add more fonts"
        )
    return layouts


def save_layouts(layouts, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(layouts, handle, ensure_ascii=False, indent=2)


def load_layouts(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def clean_and_check_text(text):

    trans_table = str.maketrans("“”‘’—–\n\t\r", "\"\"''--   ")
    text = text.translate(trans_table)
    clean_text = " ".join(text.split())
    if all(char in ALLOWED_CHARS for char in clean_text):
        return clean_text
    return None

def get_text_lines(text, font, max_width):


    lines = []
    current_line = ""
    width = 0.0
    for char in text:
        char_width = font.getlength(char)
        if current_line and width + char_width > max_width:
            lines.append(current_line)
            current_line = char
            width = char_width
        else:
            current_line += char
            width += char_width
    if current_line:
        lines.append(current_line)
    return lines

_FONT_CACHE = {}


def load_font(font_path, font_size):
    key = (font_path, font_size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(font_path, font_size)
    return _FONT_CACHE[key]


def layout_text(text, font_path, font_size, image_width, margin, line_spacing):

    font = load_font(font_path, font_size)
    available_width = image_width - 2 * margin
    lines = get_text_lines(text, font, available_width)
    line_height = int(font_size * line_spacing)
    required_height = max(len(lines) * line_height + 2 * margin, font_size * 2)
    return font, lines, line_height, required_height, available_width


def create_text_image(text, font_path, font_size, image_width, margin, line_spacing,
                      align, canvas_height=None):
    try:
        font, lines, line_height, required_height, available_width = layout_text(
            text, font_path, font_size, image_width, margin, line_spacing
        )
    except OSError:
        return None


    total_height = canvas_height if canvas_height else required_height

    image = Image.new('RGB', (image_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    y = margin
    for line in lines:
        bbox = font.getbbox(line)
        line_w = bbox[2] if bbox else 0
        if align == "center":
            x = margin + max(0, (available_width - line_w) // 2)
        elif align == "right":
            x = margin + max(0, available_width - line_w)
        else:
            x = margin
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_height

    return image

def is_font_valid_for_english(font_path):
    try:
        font = ImageFont.truetype(font_path, 20)
        box_a = font.getbbox("a")
        box_W = font.getbbox("W")
        box_comma = font.getbbox(",")
        if not box_a or not box_W or not box_comma:
            return False

        if box_a == box_W == box_comma:
            return False
        return True
    except Exception:
        return False

def get_few_valid_fonts(required_count=10):
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "~/.fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        os.path.expanduser("~/Library/Fonts"),
    ]
    valid_fonts = []
    print(f"detecting usable fonts (target {required_count})")
    for d in search_dirs:
        path = os.path.expanduser(d)
        if not os.path.exists(path):
            continue
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.ttf', '.ttc', '.otf')):
                    if any(x in f.lower() for x in ["emoji", "math", "symbol", "dingbat"]):
                        continue
                    font_path = os.path.join(root, f)
                    if is_font_valid_for_english(font_path):
                        valid_fonts.append(font_path)
                        print(f"  [+] font available: {f}")
                        if len(valid_fonts) >= required_count:
                            return valid_fonts
    return valid_fonts

def run_dataset_generation(input_parquet, base_output_dir="dataset", total_goal=5,
                           variations=10, seed=None, shuffle=False, start_index=0,
                           manifest_output=None, layout_grid=0, layouts_file=None,
                           min_chars=300, max_chars=600, canvas_height=0,
                           fixed_length=0):
    if seed is not None:
        random.seed(seed)

    fonts = get_few_valid_fonts(required_count=20)
    if not fonts:
        sys.exit("no installed font can render the Latin alphabet")

    try:
        df = pd.read_parquet(input_parquet)
    except Exception as e:
        sys.exit(f"cannot read the parquet file: {e}")

    col = 'text' if 'text' in df.columns else df.columns[0]
    all_texts = []
    too_short = 0
    too_long = 0


    pool_cap = max(5000, (start_index + total_goal) * 3 + 500)
    for t in df[col].dropna():
        clean_text = clean_and_check_text(t)
        if not clean_text:
            continue
        if len(clean_text) < min_chars:
            too_short += 1
            continue
        if max_chars and len(clean_text) > max_chars:
            too_long += 1
            continue
        if fixed_length:


            if len(clean_text) < fixed_length:
                too_short += 1
                continue
            cut = clean_text[:fixed_length]
            if " " in cut:
                cut = cut[: cut.rfind(" ")]
            clean_text = cut
        all_texts.append(clean_text)
        if len(all_texts) >= pool_cap:
            break

    print(
        f"Text filter: kept {len(all_texts)} rows in "
        f"[{min_chars}, {max_chars if max_chars else 'inf'}] characters "
        f"(rejected {too_short} too short, {too_long} too long)"
    )

    if not all_texts:
        sys.exit("no ASCII text of the requested length in the corpus")

    if shuffle:
        random.shuffle(all_texts)

    os.makedirs(base_output_dir, exist_ok=True)
    print(f"\n{len(fonts)} usable fonts; rendering {total_goal} documents...")

    p_sizes = [16, 18, 22, 24, 28]
    p_widths = [600, 750, 800, 900]
    p_margins = [20, 30, 45, 60]
    p_spacings = [1.1, 1.3, 1.5, 1.6]
    p_aligns = ["left", "center", "right"]
    layout_pool = None
    if layout_grid and layout_grid > 0:
        if layouts_file and os.path.exists(layouts_file):
            layout_pool = load_layouts(layouts_file)
            print(f"Reusing {len(layout_pool)} fixed layouts from {layouts_file}")
        else:
            layout_pool = build_layout_grid(fonts, layout_grid, seed)
            if layouts_file:
                save_layouts(layout_pool, layouts_file)
                print(f"Saved {len(layout_pool)} fixed layouts to {layouts_file}")
        variations = len(layout_pool)
        print(
            f"Layout grid mode: {total_goal} contents x {variations} fixed layouts "
            "-> both factors are pairable"
        )

    manifest_samples = []
    layout_records = []

    if canvas_height and layout_pool:
        usable = []
        rejected_fit = 0
        for text in all_texts:
            fits = True
            for layout in layout_pool:
                try:
                    _, _, _, required, _ = layout_text(
                        text,
                        layout["font_path"],
                        layout["font_size"],
                        layout["image_width"],
                        layout["margin"],
                        layout["line_spacing"],
                    )
                except OSError:
                    fits = False
                    break
                if required > canvas_height:
                    fits = False
                    break
            if fits:
                usable.append(text)
                if len(usable) >= start_index + total_goal:
                    break
            else:
                rejected_fit += 1
        print(
            f"Canvas fit: {len(usable)} texts fit in {canvas_height}px under every "
            f"layout ({rejected_fit} rejected for overflow)"
        )
        if not usable:
            raise RuntimeError(
                "no text fits the canvas under every layout; lower --min-chars, "
                "raise --canvas-height, or narrow the layout grid"
            )
        all_texts = usable

    for i in range(total_goal):
        text = all_texts[(start_index + i) % len(all_texts)]
        prefix = f"sample_{start_index + i + 1:06d}"
        folder = os.path.join(base_output_dir, prefix)
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, f"{prefix}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(base_output_dir, f"{prefix}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        v_count = 0
        image_paths = []
        attempts = 0
        while v_count < variations and attempts < 20:
            attempts += 1
            if layout_pool is not None:
                layout = layout_pool[v_count % len(layout_pool)]
            else:
                layout = {
                    "font_path": random.choice(fonts),
                    "font_size": random.choice(p_sizes),
                    "image_width": random.choice(p_widths),
                    "margin": random.choice(p_margins),
                    "line_spacing": random.choice(p_spacings),
                    "align": random.choice(p_aligns),
                }
            img = create_text_image(
                text=text, canvas_height=canvas_height or None, **layout
            )
            if img is None:
                continue
            image_path = os.path.join(folder, f"var_{v_count + 1:02d}.png")
            img.save(image_path)
            image_paths.append(image_path)
            layout_records.append(
                {
                    "sample_id": prefix,
                    "text": text,
                    "layout_id": v_count,
                    "layout": layout,
                    "path": image_path,
                }
            )
            v_count += 1

        manifest_samples.append(
            {
                "sample_id": prefix,
                "text": text,
                "images": image_paths,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{total_goal}")

        if (i + 1) == total_goal:
            layout_manifest_path = os.path.join(
                base_output_dir, "layout_manifest.json"
            )
            with open(layout_manifest_path, "w", encoding="utf-8") as f:
                json.dump(layout_records, f, ensure_ascii=False, indent=2)
            print(f"Saved layout manifest to {layout_manifest_path}")

        if manifest_output and (i + 1) == total_goal:
            manifest_parent = os.path.dirname(manifest_output)
            if manifest_parent:
                os.makedirs(manifest_parent, exist_ok=True)
            with open(manifest_output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_output_dir": base_output_dir,
                        "start_index": start_index,
                        "total_goal": total_goal,
                        "variations": variations,
                        "samples": manifest_samples,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved manifest to {manifest_output}")

    print(f"\ndataset written to {base_output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a text corpus into document images."
    )
    parser.add_argument("input_parquet", help="input parquet file")
    parser.add_argument("-o", "--output-dir", default="dataset", help="output directory")
    parser.add_argument("-n", "--num-samples", type=int, default=10000, help="number of documents")
    parser.add_argument("-v", "--variations", type=int, default=10, help="renderings per document")
    parser.add_argument("--start-index", type=int, default=0, help="index of the first corpus document, for disjoint splits")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the corpus before rendering")

    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional JSON manifest output path.",
    )
    parser.add_argument(
        "--layout-grid",
        type=int,
        default=0,
        help=(
            "If > 0, render every content under this many FIXED layouts instead "
            "of drawing a random layout per variation. This makes both factors "
            "(same content/different layout, same layout/different content) "
            "pairable, which the random mode cannot do."
        ),
    )
    parser.add_argument(
        "--layouts-file",
        default=None,
        help=(
            "JSON file for the fixed layout tuples. If it already exists it is "
            "reused, so several datasets share exactly the same layouts."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help=(
            "Minimum characters of the cleaned text. The corpus has a median of "
            "331 and a 25th percentile of 30, so the old 50-char floor produced "
            "single-sentence documents."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=600,
        help="Maximum characters; 0 disables the upper bound.",
    )
    parser.add_argument(
        "--canvas-height",
        type=int,
        default=0,
        help=(
            "Fixed image height in pixels. 0 keeps the legacy behaviour of sizing "
            "each image to its text, which leaks the content factor into image "
            "size. A fixed height equalises all images; texts that would overflow "
            "under any layout are dropped instead of clipped."
        ),
    )
    parser.add_argument(
        "--fixed-length",
        type=int,
        default=0,
        help=(
            "Truncate every text to exactly this many characters. Use it to test "
            "whether a content/layout margin comes from the meaning of the text or "
            "merely from its length. 0 disables it."
        ),
    )
    args = parser.parse_args()
    run_dataset_generation(
        args.input_parquet,
        base_output_dir=args.output_dir,
        total_goal=args.num_samples,
        variations=args.variations,
        seed=args.seed,
        shuffle=args.shuffle,
        start_index=args.start_index,
        manifest_output=args.manifest_output,
        layout_grid=args.layout_grid,
        layouts_file=args.layouts_file,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        canvas_height=args.canvas_height,
        fixed_length=args.fixed_length,
    )
