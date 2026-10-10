"""Generate every fake file in /fakes so that they are mutually consistent.

Run from the repo root:  python fakes/generate_fakes.py
Deterministic (fixed seed): re-running produces the same files. The output is committed, so you
only need to run this if you change the generator.

What it draws: a light field of view inside a dark circular border, a 5 x 4 grid of Malassez
rectangles, and small round "cells" (normal) or darker elongated ones (abnormal). Some cells sit
on purpose on grid lines to exercise the border rule.

Because the generator places every cell itself, it knows which ones are counted. It is NOT an
implementation of the border rule (that is M2's job). The placement categories are:
  inside          box strictly inside a rectangle                      -> counted
  top_left_line   centre on the outer top or left grid line            -> counted
  interior_line   centre on a grid line shared by two rectangles       -> counted (once)
  bottom_right    centre on the outer bottom or right grid line        -> NOT counted
  outside         box outside the grid, inside the circle              -> NOT counted
No cell is placed near a line intersection, so no corner cases are generated.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 20261010
ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
JSON_DIR = ROOT / "json"

SIZE = 1024          # output image is SIZE x SIZE RGB
SS = 2               # supersampling factor for anti-aliased drawing
CENTER = SIZE // 2
FIELD_RADIUS = 490   # radius of the bright field of view
COLS, ROWS = 5, 4    # 20 rectangles
RECT_W, RECT_H = 150, 120
GX0 = CENTER - COLS * RECT_W // 2    # left outer grid line
GY0 = CENTER - ROWS * RECT_H // 2    # top outer grid line
GX1 = GX0 + COLS * RECT_W            # right outer grid line
GY1 = GY0 + ROWS * RECT_H            # bottom outer grid line
LINE_PX = 3
CLEAR = 4            # min gap between a non-line cell's box and any grid line

# name, inside cells per rectangle (min, max), abnormal rate, dilution used in result.json
GOOD_IMAGES = [
    ("malassez_01", (6, 12), 0.06, 10),
    ("malassez_02", (10, 16), 0.04, 20),
    ("malassez_03", (3, 8), 0.10, 5),
]
EDGE_CELLS = {"top_left_line": 4, "interior_line": 6, "bottom_right": 4}
OUTSIDE_CELLS = (15, 25)
COUNTED = {"inside", "top_left_line", "interior_line"}


def rectangles() -> list[dict]:
    return [{"id": r * COLS + c, "x": GX0 + c * RECT_W, "y": GY0 + r * RECT_H, "w": RECT_W, "h": RECT_H}
            for r in range(ROWS) for c in range(COLS)]


def cell_shape(rng: random.Random, abnormal: bool) -> tuple[int, int]:
    """Half-width and half-height of a cell's box."""
    if not abnormal:
        r = rng.randint(7, 9)
        return r, r
    long, short = rng.randint(10, 12), rng.randint(5, 7)
    return (long, short) if rng.random() < 0.5 else (short, long)


def place_cells(rng: random.Random, per_rect: tuple[int, int], abnormal_rate: float) -> list[dict]:
    cells: list[dict] = []

    def free(cx, cy, rx, ry) -> bool:
        return all(abs(cx - o["cx"]) > rx + o["rx"] + 3 or abs(cy - o["cy"]) > ry + o["ry"] + 3 for o in cells)

    def add(kind, sampler, n):
        placed = 0
        while placed < n:
            abnormal = rng.random() < abnormal_rate
            rx, ry = cell_shape(rng, abnormal)
            for _ in range(200):
                cx, cy = sampler(rx, ry)
                if free(cx, cy, rx, ry):
                    cells.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry, "abnormal": abnormal, "kind": kind})
                    placed += 1
                    break
            else:
                raise RuntimeError(f"could not place a {kind} cell")

    def along(lo, hi, r):
        # a position along a grid line segment, away from the intersections at lo and hi
        return rng.randint(lo + r + CLEAR + 2, hi - r - CLEAR - 2)

    for rect in rectangles():
        def inside(rx, ry, rect=rect):
            return (rng.randint(rect["x"] + rx + CLEAR, rect["x"] + rect["w"] - rx - CLEAR),
                    rng.randint(rect["y"] + ry + CLEAR, rect["y"] + rect["h"] - ry - CLEAR))
        add("inside", inside, rng.randint(*per_rect))

    def on_top_left(rx, ry):
        if rng.random() < 0.5:   # outer top line
            c = rng.randrange(COLS)
            return along(GX0 + c * RECT_W, GX0 + (c + 1) * RECT_W, rx), GY0
        r = rng.randrange(ROWS)  # outer left line
        return GX0, along(GY0 + r * RECT_H, GY0 + (r + 1) * RECT_H, ry)

    def on_interior(rx, ry):
        if rng.random() < 0.5:   # vertical interior line
            x = GX0 + rng.randint(1, COLS - 1) * RECT_W
            r = rng.randrange(ROWS)
            return x, along(GY0 + r * RECT_H, GY0 + (r + 1) * RECT_H, ry)
        y = GY0 + rng.randint(1, ROWS - 1) * RECT_H
        c = rng.randrange(COLS)
        return along(GX0 + c * RECT_W, GX0 + (c + 1) * RECT_W, rx), y

    def on_bottom_right(rx, ry):
        if rng.random() < 0.5:   # outer bottom line
            c = rng.randrange(COLS)
            return along(GX0 + c * RECT_W, GX0 + (c + 1) * RECT_W, rx), GY1
        r = rng.randrange(ROWS)  # outer right line
        return GX1, along(GY0 + r * RECT_H, GY0 + (r + 1) * RECT_H, ry)

    def outside(rx, ry):
        while True:
            cx, cy = rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1)
            in_field = (cx - CENTER) ** 2 + (cy - CENTER) ** 2 < (FIELD_RADIUS - max(rx, ry) - 12) ** 2
            off_grid = (cx + rx < GX0 - CLEAR or cx - rx > GX1 + CLEAR or
                        cy + ry < GY0 - CLEAR or cy - ry > GY1 + CLEAR)
            if in_field and off_grid:
                return cx, cy

    add("top_left_line", on_top_left, EDGE_CELLS["top_left_line"])
    add("interior_line", on_interior, EDGE_CELLS["interior_line"])
    add("bottom_right", on_bottom_right, EDGE_CELLS["bottom_right"])
    add("outside", outside, rng.randint(*OUTSIDE_CELLS))
    cells.sort(key=lambda c: (c["cy"] - c["ry"], c["cx"] - c["rx"]))
    return cells


def render(cells: list[dict], np_rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (SIZE * SS, SIZE * SS), (236, 232, 226))
    d = ImageDraw.Draw(img)
    s = SS
    for i in range(COLS + 1):
        x = GX0 + i * RECT_W
        d.line([(x * s, GY0 * s), (x * s, GY1 * s)], fill=(70, 70, 80), width=LINE_PX * s)
    for j in range(ROWS + 1):
        y = GY0 + j * RECT_H
        d.line([(GX0 * s, y * s), (GX1 * s, y * s)], fill=(70, 70, 80), width=LINE_PX * s)
    for c in cells:
        box = [(c["cx"] - c["rx"]) * s, (c["cy"] - c["ry"]) * s, (c["cx"] + c["rx"]) * s, (c["cy"] + c["ry"]) * s]
        if c["abnormal"]:
            d.ellipse(box, fill=(95, 70, 112), outline=(55, 35, 75), width=2 * s)
        else:
            d.ellipse(box, fill=(160, 150, 185), outline=(90, 80, 122), width=2 * s)
            k = 0.45
            d.ellipse([(c["cx"] - k * c["rx"]) * s, (c["cy"] - k * c["ry"]) * s,
                       (c["cx"] + k * c["rx"]) * s, (c["cy"] + k * c["ry"]) * s], fill=(200, 194, 215))
    img = img.resize((SIZE, SIZE), Image.LANCZOS)

    a = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((xx - CENTER) ** 2 + (yy - CENTER) ** 2)
    a *= (1.0 - 0.10 * (dist / FIELD_RADIUS) ** 2)[..., None]                      # vignette
    a += np_rng.normal(0, 3.0, a.shape)                                            # sensor noise
    edge = np.clip((FIELD_RADIUS - dist) / 6.0, 0, 1)[..., None]                   # soft circular border
    a = a * edge + np.array([30, 30, 35], np.float32) * (1 - edge)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def result_from(cells: list[dict], dets: list[dict], clss: list[dict], dilution: float) -> dict:
    counted = [(d, c) for cell, d, c in zip(cells, dets, clss) if cell["kind"] in COUNTED]
    rc = [{"x": d["x"] + d["w"] // 2, "y": d["y"] + d["h"] // 2, "label": c["label"], "score": c["score"]}
          for d, c in counted]
    total = len(rc)
    n_rect = len(rectangles())
    return {
        "cells": rc,
        "rectangles_counted": n_rect,
        "total_count": total,
        "abnormal_count": sum(c["label"] == "abnormal" for c in rc),
        "dilution_factor": dilution,
        "concentration_per_uL": round(total / n_rect * 100 * dilution, 2),
        "confidence": round(sum(c["score"] for c in rc) / total, 2) if total else 0,
    }


def make_samples(rng: random.Random, results: dict[str, dict]) -> list[dict]:
    """Fake cases-service records (sample_detail.schema.json) covering every status."""
    def uid() -> str:
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))

    def image_id() -> str:
        return "img_" + "".join(rng.choice("0123456789abcdef") for _ in range(12))

    def record(sample_id, ts, source, dilution):
        res = results[source]
        img = {"id": image_id(), "sample_id": sample_id, "file_path": None, "quality_ok": True,
               "rejection_reason": None, "uploaded_at": ts}
        rec = {"id": uid(), "image_id": img["id"], "total_count": res["total_count"],
               "abnormal_count": res["abnormal_count"], "rectangles_counted": res["rectangles_counted"],
               "concentration": round(res["total_count"] / res["rectangles_counted"] * 100 * dilution, 2),
               "confidence": res["confidence"], "cells": res["cells"], "model_version": "fake-0.1"}
        return img, rec

    # (code, status, technician, source image or None, dilution, correction, validated_at, comment)
    plan = [
        ("S-20261010-001", "pending", "tech_01", None, 10, None, None, None),
        ("S-20261010-002", "pending", "tech_01", "malassez_01", 10, None, None, None),
        ("S-20261010-003", "corrected", "tech_02", "malassez_02", 20, (3, 1), None, "Three cells missed near debris."),
        ("S-20261010-004", "validated", "tech_02", "malassez_03", 5, None, "2026-10-10T11:40:00Z", None),
        ("S-20261010-005", "validated", "tech_01", "malassez_01", 10, (-2, 0), "2026-10-10T12:05:00Z",
         "Two artefacts counted as cells."),
    ]
    out = []
    for i, (code, status, tech, source, dil, corr, validated_at, comment) in enumerate(plan):
        sid = uid()
        created = f"2026-10-10T{9 + i:02d}:00:00Z"
        sample = {"id": sid, "code": code, "created_at": created, "dilution_factor": dil,
                  "technician": tech, "status": status}
        images, result, validation = [], None, None
        if source:
            img, result = record(sid, f"2026-10-10T{9 + i:02d}:05:00Z", source, dil)
            images.append(img)
        if status != "pending":
            validation = {"id": uid(), "sample_id": sid,
                          "corrected_count": result["total_count"] + corr[0] if corr else None,
                          "corrected_abnormal": result["abnormal_count"] + corr[1] if corr else None,
                          "validator": "expert_01", "comment": comment, "validated_at": validated_at}
        out.append({"sample": sample, "images": images, "result": result, "validation": validation})
    return out


REJECTIONS = [
    {"accepted": False, "reason": "blurry", "detail": "Focus score 12.4 is below the 40.0 threshold."},
    {"accepted": False, "reason": "too_dark", "detail": "Mean brightness 31 is below the 60 threshold."},
    {"accepted": False, "reason": "overexposed", "detail": "41% of pixels are saturated (max 5%)."},
]


def main() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    IMG_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    for name, per_rect, abnormal_rate, dilution in GOOD_IMAGES:
        cells = place_cells(rng, per_rect, abnormal_rate)
        img = render(cells, np_rng)
        img.save(IMG_DIR / f"{name}.png")

        dets = [{"x": c["cx"] - c["rx"], "y": c["cy"] - c["ry"], "w": 2 * c["rx"], "h": 2 * c["ry"],
                 "score": round(rng.uniform(0.80, 0.99), 2)} for c in cells]
        clss = [{"label": "abnormal", "score": round(rng.uniform(0.65, 0.95), 2)} if c["abnormal"]
                else {"label": "normal", "score": round(rng.uniform(0.80, 0.99), 2)} for c in cells]
        result = result_from(cells, dets, clss, dilution)
        results[name] = result

        write_json(JSON_DIR / f"{name}.preprocess.json",
                   {"image": f"fakes/images/{name}.png", "rectangles": rectangles()})
        write_json(JSON_DIR / f"{name}.detections.json", dets)
        write_json(JSON_DIR / f"{name}.classifications.json", clss)
        write_json(JSON_DIR / f"{name}.result.json", result)
        kinds = {k: sum(c["kind"] == k for c in cells) for k in
                 ("inside", "top_left_line", "interior_line", "bottom_right", "outside")}
        print(f"{name}: {len(cells)} cells drawn {kinds} -> counted {result['total_count']}, "
              f"abnormal {result['abnormal_count']}, {result['concentration_per_uL']} /uL")

        if name == "malassez_01":  # quality-rejection test images derived from the first good image
            img.filter(ImageFilter.GaussianBlur(6)).save(IMG_DIR / "malassez_blurry.png")
            dark = (np.asarray(img).astype(np.float32) * 0.18).astype(np.uint8)
            Image.fromarray(dark, "RGB").save(IMG_DIR / "malassez_dark.png")

    write_json(JSON_DIR / "samples.json", make_samples(rng, results))
    write_json(JSON_DIR / "rejection_examples.json", REJECTIONS)
    print(f"wrote {IMG_DIR} and {JSON_DIR}")


if __name__ == "__main__":
    main()
