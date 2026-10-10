"""Contract tests for classify_cell, run with an untrained checkpoint so no real weights are needed.

    python -m pytest ai/classification/tests

They check the output shape, not accuracy. The crops come from fakes/crops (code testing only).
"""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from ai.classification import classifier as clf_module
from ai.classification.model import build_model, save_checkpoint

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "contracts" / "classification.schema.json").read_text(encoding="utf-8"))
CROPS = sorted((ROOT / "fakes" / "crops").glob("*/*.png"))[:16]


@pytest.fixture(autouse=True)
def untrained_weights(tmp_path, monkeypatch):
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, build_model("resnet18", pretrained=False), "resnet18", 224,
                    {0: "abnormal", 1: "normal"})
    monkeypatch.setenv("HACKCURE_CLASSIFIER_WEIGHTS", str(path))
    monkeypatch.setattr(clf_module, "_classifier", None)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def test_output_matches_contract_schema():
    for path in CROPS:
        jsonschema.validate(clf_module.classify_cell(load(path)), SCHEMA)


@pytest.mark.parametrize("shape", [(12, 12, 3), (7, 15, 3), (224, 224, 3), (1, 1, 3)])
def test_any_crop_size(shape):
    """The contract crop is the exact box image[y:y+h, x:x+w], so sizes vary."""
    crop = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    jsonschema.validate(clf_module.classify_cell(crop), SCHEMA)


def test_score_is_probability_of_returned_label():
    clf = clf_module.get_classifier()
    p = clf.predict_proba([load(CROPS[0])])[0]
    out = clf_module.classify_cell(load(CROPS[0]))
    expected = p if out["label"] == "abnormal" else 1 - p
    assert out["score"] == pytest.approx(expected, abs=1e-3)
    assert out["score"] >= 0.5 or clf.threshold < 0.5


def test_batch_equals_single():
    crops = [load(p) for p in CROPS]
    batch = clf_module.classify_cells(crops)
    single = [clf_module.classify_cell(c) for c in crops]
    assert [b["label"] for b in batch] == [s["label"] for s in single]
    assert [b["score"] for b in batch] == pytest.approx([s["score"] for s in single], abs=1e-3)
    assert clf_module.classify_cells([]) == []
