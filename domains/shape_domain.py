import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw


SHAPES = ["circle", "square", "triangle", "star", "hexagon", "cross"]
COLOURS = [
    (220, 40, 40), (40, 120, 220), (40, 170, 90),
    (230, 170, 30), (150, 60, 200), (30, 30, 30),
]
TEXTURES = ["noise", "checker", "stripes", "gradient", "blobs"]


def background(kind, size, rng):

    if kind == "noise":
        array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(array)
    if kind == "checker":
        block = int(rng.integers(8, 32))
        yy, xx = np.indices((size, size))
        mask = ((yy // block) + (xx // block)) % 2 == 0
        base = np.empty((size, size, 3), dtype=np.uint8)
        base[mask] = rng.integers(120, 256, 3)
        base[~mask] = rng.integers(0, 120, 3)
        return Image.fromarray(base)
    if kind == "stripes":
        width = int(rng.integers(4, 20))
        yy, _ = np.indices((size, size))
        mask = (yy // width) % 2 == 0
        base = np.empty((size, size, 3), dtype=np.uint8)
        base[mask] = rng.integers(140, 256, 3)
        base[~mask] = rng.integers(0, 130, 3)
        return Image.fromarray(base)
    if kind == "gradient":
        ramp = np.linspace(0, 255, size, dtype=np.uint8)
        base = np.repeat(ramp[:, None], size, axis=1)[:, :, None]
        tint = np.array([rng.integers(60, 256) for _ in range(3)], dtype=np.float32) / 255
        array = np.clip(base.astype(np.float32) * tint, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    array = np.full((size, size, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(4, 10))):
        x, y = rng.integers(0, size, 2)
        r = int(rng.integers(10, 45))
        colour = tuple(int(v) for v in rng.integers(0, 256, 3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
    return image


def draw_shape(draw, shape, centre, radius, colour):
    x, y = centre
    r = radius
    if shape == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
    elif shape == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=colour)
    elif shape == "triangle":
        draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=colour)
    elif shape == "star":
        points = []
        for index in range(10):
            angle = -np.pi / 2 + index * np.pi / 5
            radius = r if index % 2 == 0 else r * 0.45
            points.append((x + radius * np.cos(angle), y + radius * np.sin(angle)))
        draw.polygon(points, fill=colour)
    elif shape == "hexagon":
        points = [
            (x + r * np.cos(a), y + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]
        draw.polygon(points, fill=colour)
    else:
        arm = r * 0.35
        draw.rectangle([x - arm, y - r, x + arm, y + r], fill=colour)
        draw.rectangle([x - r, y - arm, x + r, y + arm], fill=colour)


def nuisances(count, seed):

    rng = np.random.default_rng(seed)
    seen = set()
    out = []
    while len(out) < count:
        item = {
            "texture": TEXTURES[len(out) % len(TEXTURES)],
            "rotation": int(rng.integers(0, 360)),
            "scale": round(float(rng.uniform(0.45, 0.75)), 3),
            "dx": int(rng.integers(-25, 26)),
            "dy": int(rng.integers(-25, 26)),
        }
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def render(shape, colour, nuisance, size, seed):
    rng = np.random.default_rng(seed)
    image = background(nuisance["texture"], size, rng)

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    centre = (size // 2 + nuisance["dx"], size // 2 + nuisance["dy"])
    draw_shape(draw, shape, centre, int(size * nuisance["scale"]), colour + (255,))
    layer = layer.rotate(nuisance["rotation"], resample=Image.BICUBIC, center=centre)
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    return image


def main():
    parser = argparse.ArgumentParser(description="Shapes-on-texture grid.")
    parser.add_argument("--contents", type=int, default=16)
    parser.add_argument("--nuisances", type=int, default=8)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="dataset/shape_grid")
    args = parser.parse_args()

    content_list = []
    for shape in SHAPES:
        for colour in COLOURS:
            content_list.append((shape, colour))
            if len(content_list) >= args.contents:
                break
        if len(content_list) >= args.contents:
            break

    nuisance_list = nuisances(args.nuisances, args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for content_id, (shape, colour) in enumerate(content_list):
        for nuisance_id, nuisance in enumerate(nuisance_list):
            image = render(shape, colour, nuisance, args.size, args.seed + content_id)
            folder = os.path.join(args.out, f"c{content_id:03d}")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"l{nuisance_id:02d}.png")
            image.save(path)
            manifest.append(
                {
                    "topic": shape,
                    "document": f"c{content_id:03d}",
                    "layout": nuisance_id,
                    "path": path,
                }
            )
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"{len(content_list)} contents x {len(nuisance_list)} nuisances = "
        f"{len(manifest)} images in {args.out}"
    )


if __name__ == "__main__":
    main()
