"""Internal Python interfaces of the AI pipeline: the source of truth for M1, M2, M3 and M4.

Signatures and docstrings only. Real implementations live in /ai/<stage>/ and must keep these
exact signatures. Fake implementations that return /fakes data live in fakes/stubs.py.

Pipeline (wired by M4 in services/inference):

    pre = preprocess(image)                                   # M1
    dets = detect_cells(pre["image"])                         # M2
    cells = [{"detection": d,
              "classification": classify_cell(crop(pre["image"], d))}   # M3
             for d in dets]
    result = compute_result(cells, pre["rectangles"], dilution)          # M2

Conventions:
- Images are np.ndarray of shape (H, W, 3), dtype uint8, RGB channel order.
- Every coordinate is an int pixel in the frame of the image returned by preprocess()
  (PROPOSED — confirm with team). preprocess() may crop or resize, but everything downstream
  uses its output frame.
- A crop for classify_cell is image[y:y+h, x:x+w] for a detection box.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np


class Rectangle(TypedDict):
    """contracts/rectangle.schema.json"""
    id: int
    x: int
    y: int
    w: int
    h: int


class PreprocessOutput(TypedDict):
    """In-memory form of contracts/preprocess_output.schema.json (there 'image' is a file path)."""
    image: np.ndarray
    rectangles: list[Rectangle]


class Detection(TypedDict):
    """contracts/detection.schema.json"""
    x: int
    y: int
    w: int
    h: int
    score: float


class Classification(TypedDict):
    """contracts/classification.schema.json"""
    label: Literal["normal", "abnormal"]
    score: float


class ClassifiedCell(TypedDict):
    """One item of contracts/classified_cells.schema.json (PROPOSED — confirm with team)."""
    detection: Detection
    classification: Classification


class ResultCell(TypedDict):
    """contracts/result.schema.json#/$defs/cell"""
    x: int
    y: int
    label: Literal["normal", "abnormal"]
    score: float


class AnalysisResult(TypedDict):
    """contracts/result.schema.json"""
    cells: list[ResultCell]
    rectangles_counted: int
    total_count: int
    abnormal_count: int
    dilution_factor: float
    concentration_per_uL: float
    confidence: float


def preprocess(image: np.ndarray) -> PreprocessOutput:
    """M1 (Isra). Clean the raw microscope image and locate the Malassez rectangles to count.

    Args:
        image: raw RGB image, (H, W, 3) uint8.

    Returns:
        {"image": cleaned RGB image (H', W', 3) uint8,
         "rectangles": [{"id", "x", "y", "w", "h"}, ...]}: every rectangle to count, ids unique.
        (x, y) is where the rectangle's top and left grid lines cross.

    Raises:
        ValueError: if no grid can be found. M4 maps this to an error response.
    """
    raise NotImplementedError


def detect_cells(image: np.ndarray) -> list[Detection]:
    """M2 (Imene). Detect every cell in the preprocessed image with YOLOv8.

    Args:
        image: the "image" returned by preprocess().

    Returns:
        One {"x", "y", "w", "h", "score"} box per cell, anywhere in the image, before the border
        rule. (x, y) is the box's top-left corner; score is in [0, 1].
    """
    raise NotImplementedError


def classify_cell(crop: np.ndarray) -> Classification:
    """M3 (Sarah). Classify one cell crop as normal or abnormal.

    Args:
        crop: image[y:y+h, x:x+w] for one detection, RGB uint8. Resizing is up to the classifier.

    Returns:
        {"label": "normal" | "abnormal", "score": probability of that label in [0, 1]}.
        "abnormal" is a flag for expert review, not a diagnosis.
    """
    raise NotImplementedError


def compute_result(cells: list[ClassifiedCell], rectangles: list[Rectangle], dilution: float) -> AnalysisResult:
    """M2 (Imene). Apply the border rule, count, and compute the concentration.

    Args:
        cells: every detection paired with its classification (PROPOSED shape — confirm with team).
        rectangles: from preprocess(); all of them are counted.
        dilution: the sample's dilution factor (> 0).

    Returns:
        A result JSON (contracts/result.schema.json) where:
        - cells holds only the counted cells, as box centres (x + w // 2, y + h // 2)
        - rectangles_counted = len(rectangles)
        - total_count = len(cells); abnormal_count = number labelled "abnormal"
        - concentration_per_uL = round(total_count / rectangles_counted * 100 * dilution, 2)
        - confidence = round(mean(cell scores), 2), or 0 if there are no cells (PROPOSED)

    Border rule (PROPOSED — confirm with the lab protocol): a cell is counted in a rectangle if its
    box overlaps the rectangle and does not touch or cross the rectangle's bottom or right line.
    In effect, cells on top and left lines are counted and cells on bottom and right lines are not,
    and each cell is counted at most once.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------
# Shared crop helpers. Implemented in ai/common/crop.py, listed here so every stage sees the same
# signatures. `cell` is a Detection; `labels[i]` is the {"label": ...} of `cells[i]` (order-aligned,
# since detections have no id field; a cell's id is its index).
#
# NOTE: classify_cell's contract input is still the exact box crop image[y:y+h, x:x+w]. Switching
# it to crop_cell() output is an open question (PROPOSED, see README "Fake data for M3") and needs
# a contract PR.
# ---------------------------------------------------------------------------------------------

def crop_cell(image: np.ndarray, cell: Detection, margin: float = 0.25, size: int = 224) -> np.ndarray:
    """Square crop centred on the cell's box, side = max(w, h) * (1 + 2 * margin), replicate-padded
    at image edges, resized to (size, size, 3) RGB uint8."""
    raise NotImplementedError


def export_crops(image: np.ndarray, cells: list[Detection], labels: list[dict], out_dir: str) -> str:
    """Save crop_cell() of every cell to out_dir/<label>/cell_<id:04d>.png (ImageFolder layout) and
    write out_dir/labels.csv (crop_path, cell_id, label, x, y, w, h). Returns the labels.csv path."""
    raise NotImplementedError


def draw_labeled_overlay(image: np.ndarray, cells: list[Detection], labels: list[dict]) -> np.ndarray:
    """Copy of the image with every box drawn: green = normal, red = abnormal."""
    raise NotImplementedError
