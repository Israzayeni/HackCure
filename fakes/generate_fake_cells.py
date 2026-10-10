"""Generate the M3 (classification) fake data: one image of labelled cells, plus its crops.

Run from the repo root:  python fakes/generate_fake_cells.py
Deterministic (fixed seed). The output is committed, so M3 does not need to run this.

Writes:
  fakes/images/fake_cells.png            image as if it came out of preprocess() (RGB)
  fakes/json/fake_cells.detections.json  detect_cells() format (contracts/detections.schema.json), no labels
  fakes/json/fake_cells.labels.json      true label per detection, SAME ORDER as the detections
                                         (detections have no id field, so cell id = index)
  fakes/crops/{normal,abnormal}/cell_<id>.png + fakes/crops/labels.csv   via ai.common.crop.export_crops
  fakes/images/fake_cells_overlay.png    via ai.common.crop.draw_labeled_overlay

The drawn cells are for testing code only: never train on them or report accuracy on them.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ai.common.crop import draw_labeled_overlay, export_crops  # noqa: E402

SEED = 20261011
SIZE = 1024
SS = 2                 # supersampling for anti-aliased shapes
GRID_STEP = 128
N_NORMAL, N_ABNORMAL = 96, 24
N_NEAR_EDGE = 6        # cells placed within a few px of the image border, to exercise the crop padding
GAP = 4                # min px between two cell boxes
NAME = "fake_cells"
IMG_DIR, JSON_DIR, CROP_DIR = ROOT / "fakes" / "images", ROOT / "fakes" / "json", ROOT / "fakes" / "crops"


def normal_shape(rng: random.Random) -> list[tuple[float, float]]:
    """Small round cell: polygon relative to its centre."""
    r = rng.uniform(6, 8)
    return [(r * math.cos(t), r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]


def abnormal_shape(rng: random.Random) -> list[tuple[float, float]]:
    """Bigger irregular cell: a rotated ellipse with a bump on one side."""
    a, b = rng.uniform(12, 16), rng.uniform(7, 10)
    rot, bump_at, bump = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(4, 7)
    pts = []
    for t in np.linspace(0, 2 * math.pi, 96, endpoint=False):
        d = (t - bump_at + math.pi) % (2 * math.pi) - math.pi
        k = 1 + bump * math.exp(-d * d / (2 * 0.35 ** 2)) / math.hypot(a * math.cos(t), b * math.sin(t))
        x, y = a * math.cos(t) * k, b * math.sin(t) * k
        pts.append((x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)))
    return pts


def box_of(poly) -> tuple[int, int, int, int]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0 = math.floor(min(xs)), math.floor(min(ys))
    return x0, y0, math.ceil(max(xs)) - x0, math.ceil(max(ys)) - y0


def place(rng: random.Random) -> list[dict]:
    kinds = ["normal"] * N_NORMAL + ["abnormal"] * N_ABNORMAL
    rng.shuffle(kinds)
    cells: list[dict] = []
    for i, kind in enumerate(kinds):
        shape = normal_shape(rng) if kind == "normal" else abnormal_shape(rng)
        _, _, w, h = box_of(shape)
        for _ in range(500):
            if i < N_NEAR_EDGE:   # hug one of the four borders
                side = i % 4
                along = rng.uniform(40, SIZE - 40)
                off = rng.uniform(1, 4)
                cx, cy = [(off + w / 2, along), (SIZE - off - w / 2, along),
                          (along, off + h / 2), (along, SIZE - off - h / 2)][side]
            else:
                cx, cy = rng.uniform(20, SIZE - 20), rng.uniform(20, SIZE - 20)
            poly = [(cx + px, cy + py) for px, py in shape]
            x, y, bw, bh = box_of(poly)
            if x < 0 or y < 0 or x + bw > SIZE or y + bh > SIZE:
                continue
            if all(x + bw + GAP <= o["x"] or o["x"] + o["w"] + GAP <= x or
                   y + bh + GAP <= o["y"] or o["y"] + o["h"] + GAP <= y for o in cells):
                cells.append({"x": x, "y": y, "w": bw, "h": bh, "poly": poly, "label": kind})
                break
        else:
            raise RuntimeError("could not place a cell")
    cells.sort(key=lambda c: (c["y"], c["x"]))
    return cells


def render(cells: list[dict], np_rng: np.random.Generator) -> np.ndarray:
    img = Image.new("RGB", (SIZE * SS, SIZE * SS), (238, 235, 228))
    d = ImageDraw.Draw(img)
    for v in range(GRID_STEP, SIZE, GRID_STEP):
        d.line([(v * SS, 0), (v * SS, SIZE * SS)], fill=(150, 150, 160), width=2 * SS)
        d.line([(0, v * SS), (SIZE * SS, v * SS)], fill=(150, 150, 160), width=2 * SS)
    for c in cells:
        poly = [(x * SS, y * SS) for x, y in c["poly"]]
        if c["label"] == "normal":
            d.polygon(poly, fill=(70, 58, 95), outline=(40, 30, 60))
        else:
            d.polygon(poly, fill=(196, 120, 64), outline=(120, 64, 30))
    a = np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS)).astype(np.float32)
    a += np_rng.normal(0, 2.5, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    rng = random.Random(SEED)
    cells = place(rng)
    image = render(cells, np.random.default_rng(SEED))

    dets = [{"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"], "score": round(rng.uniform(0.6, 0.99), 2)}
            for c in cells]
    labels = [{"label": c["label"]} for c in cells]

    Image.fromarray(image).save(IMG_DIR / f"{NAME}.png")
    write_json(JSON_DIR / f"{NAME}.detections.json", dets)
    write_json(JSON_DIR / f"{NAME}.labels.json", labels)

    for label in ("normal", "abnormal"):          # drop stale crops from a previous run
        for old in (CROP_DIR / label).glob("cell_*.png"):
            old.unlink()
    export_crops(image, dets, labels, CROP_DIR)
    Image.fromarray(draw_labeled_overlay(image, dets, labels)).save(IMG_DIR / f"{NAME}_overlay.png")

    n_abn = sum(lab["label"] == "abnormal" for lab in labels)
    print(f"{NAME}: {len(dets)} cells ({len(dets) - n_abn} normal, {n_abn} abnormal) -> {CROP_DIR}")


if __name__ == "__main__":
    main()
