"""Shared cell-crop helpers (signatures also listed in contracts/interfaces.py).

Images are RGB uint8 np.ndarray (H, W, 3), as everywhere in the pipeline. Cells are detection
dicts from contracts/detection.schema.json: (x, y) = top-left corner of the box, w, h = size.
Detections have no id field, so a cell's id is its index in the detections list, and labels
are matched to cells by order (contracts/classifications.schema.json).

Uses numpy + Pillow only. If you load images with OpenCV, convert BGR -> RGB first:
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL_COLORS = {"normal": (0, 200, 0), "abnormal": (230, 0, 0)}


def crop_cell(image: np.ndarray, cell: dict, margin: float = 0.25, size: int = 224) -> np.ndarray:
    """Square crop centred on the cell's box, resized to (size, size, 3) RGB uint8.

    The side is max(w, h) * (1 + 2 * margin). Where the square goes past the image edge, the
    image is replicate-padded, so cells near the border still give a full crop.
    """
    side = max(1, round(max(cell["w"], cell["h"]) * (1 + 2 * margin)))
    cx2, cy2 = 2 * cell["x"] + cell["w"], 2 * cell["y"] + cell["h"]   # twice the centre, to stay in ints
    x0, y0 = (cx2 - side) // 2, (cy2 - side) // 2
    x1, y1 = x0 + side, y0 + side
    h, w = image.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)), (0, 0))
    window = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    square = np.pad(window, pad, mode="edge")
    return np.asarray(Image.fromarray(square).resize((size, size), Image.BILINEAR))


def export_crops(image: np.ndarray, cells: list[dict], labels: list[dict], out_dir: str | Path,
                 margin: float = 0.25, size: int = 224) -> str:
    """Save crop_cell() of every cell to out_dir/<label>/cell_<id:04d>.png and write out_dir/labels.csv.

    labels[i] is the {"label": ...} of cells[i] (same order; classifications also work). The id is
    the index in cells. labels.csv columns: crop_path (relative to out_dir), cell_id, label, x, y, w, h.
    The folder layout loads directly with torchvision.datasets.ImageFolder(out_dir).
    Returns the path of labels.csv.
    """
    if len(cells) != len(labels):
        raise ValueError(f"{len(cells)} cells but {len(labels)} labels")
    out = Path(out_dir)
    rows = []
    for cell_id, (cell, lab) in enumerate(zip(cells, labels)):
        label = lab["label"]
        rel = Path(label) / f"cell_{cell_id:04d}.png"
        (out / label).mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop_cell(image, cell, margin, size)).save(out / rel)
        rows.append([rel.as_posix(), cell_id, label, cell["x"], cell["y"], cell["w"], cell["h"]])
    csv_path = out / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["crop_path", "cell_id", "label", "x", "y", "w", "h"])
        writer.writerows(rows)
    return str(csv_path)


def draw_labeled_overlay(image: np.ndarray, cells: list[dict], labels: list[dict]) -> np.ndarray:
    """Copy of the image with every box drawn: green = normal, red = abnormal. labels[i] goes with cells[i]."""
    if len(cells) != len(labels):
        raise ValueError(f"{len(cells)} cells but {len(labels)} labels")
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cell, lab in zip(cells, labels):
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=LABEL_COLORS[lab["label"]], width=2)
    return np.asarray(img)
