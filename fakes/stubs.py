"""Fake implementations of the contracts/interfaces.py functions. They return the /fakes data, no AI.

Each fake has the exact signature of the real function, so a call site only changes its import at
integration:

    from fakes.stubs import fake_preprocess as preprocess          # until H7
    from ai.preprocessing.<module> import preprocess               # after the swap

The fakes recognize the fake images (and crops/boxes taken from them). On any other input (for
example after a real upstream stage has been swapped in) they warn and fall back to the data of
malassez_01, so the pipeline keeps running end to end during integration.

Run from the repo root (so that `fakes` is importable).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

FAKES = Path(__file__).resolve().parent
IMAGES = ["malassez_01", "malassez_02", "malassez_03"]
DEFAULT = "malassez_01"


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array (H, W, 3), the format every pipeline stage expects."""
    return np.asarray(Image.open(path).convert("RGB"))


def _json(name: str):
    return json.loads((FAKES / "json" / name).read_text(encoding="utf-8"))


def _digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha1(str(a.shape).encode() + a.tobytes()).hexdigest()


@lru_cache(maxsize=None)
def _index() -> dict:
    """Lookup tables from fake images, crops and detection sets to the fake JSON outputs."""
    by_image, by_crop, by_boxes = {}, {}, {}
    for name in IMAGES:
        img = load_image(FAKES / "images" / f"{name}.png")
        dets = _json(f"{name}.detections.json")
        clss = _json(f"{name}.classifications.json")
        by_image[_digest(img)] = name
        for d, c in zip(dets, clss):
            by_crop[_digest(img[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]])] = c
        by_boxes[tuple((d["x"], d["y"], d["w"], d["h"]) for d in dets)] = name
    return {"image": by_image, "crop": by_crop, "boxes": by_boxes}


def _which(image: np.ndarray, stage: str) -> str:
    name = _index()["image"].get(_digest(image))
    if name is None:
        warnings.warn(f"{stage}: unknown image, returning fake data of {DEFAULT}", stacklevel=3)
        return DEFAULT
    return name


def fake_preprocess(image: np.ndarray) -> dict:
    """Fake of preprocess() (M1). The fake images are already clean, so the image is returned as is."""
    name = _which(image, "fake_preprocess")
    return {"image": image, "rectangles": _json(f"{name}.preprocess.json")["rectangles"]}


def fake_detect_cells(image: np.ndarray) -> list[dict]:
    """Fake of detect_cells() (M2): the boxes of the cells drawn on the fake image."""
    return _json(f"{_which(image, 'fake_detect_cells')}.detections.json")


def fake_classify_cell(crop: np.ndarray) -> dict:
    """Fake of classify_cell() (M3): the stored label of a crop taken from a fake detection box."""
    c = _index()["crop"].get(_digest(crop))
    if c is None:
        warnings.warn("fake_classify_cell: unknown crop, returning normal/0.9", stacklevel=2)
        return {"label": "normal", "score": 0.9}
    return dict(c)


def fake_compute_result(cells: list[dict], rectangles: list[dict], dilution: float) -> dict:
    """Fake of compute_result() (M2): the stored result of the matching fake image, recomputed
    for the requested dilution. It does NOT apply the border rule, which is M2's job."""
    key = tuple((c["detection"]["x"], c["detection"]["y"], c["detection"]["w"], c["detection"]["h"])
                for c in cells)
    name = _index()["boxes"].get(key)
    if name is None:
        warnings.warn(f"fake_compute_result: unknown cells, returning fake result of {DEFAULT}", stacklevel=2)
        name = DEFAULT
    result = _json(f"{name}.result.json")
    return with_dilution(result, dilution)


def with_dilution(result: dict, dilution: float) -> dict:
    """Copy of a result JSON with dilution_factor and concentration_per_uL set for another dilution."""
    out = dict(result)
    out["dilution_factor"] = dilution
    out["concentration_per_uL"] = round(out["total_count"] / out["rectangles_counted"] * 100 * dilution, 2)
    return out


def fake_analyze(image: np.ndarray, dilution_factor: float) -> dict:
    """What M4's /analyze does, wired to the fakes. Each line maps to one member's real function."""
    pre = fake_preprocess(image)                                                # M1
    img = pre["image"]
    dets = fake_detect_cells(img)                                               # M2
    cells = [{"detection": d,
              "classification": fake_classify_cell(img[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]])}  # M3
             for d in dets]
    return fake_compute_result(cells, pre["rectangles"], dilution_factor)       # M2
