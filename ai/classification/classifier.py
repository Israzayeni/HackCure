"""classify_cell() for the pipeline (M3). Replaces fakes.stubs.fake_classify_cell at integration.

The weights are loaded once, on the first call, from the path in the environment variable
HACKCURE_CLASSIFIER_WEIGHTS (default: ai/classification/weights/best.pt). Inference runs on CPU
unless HACKCURE_DEVICE says otherwise.

"abnormal" is returned when p(abnormal) >= HACKCURE_ABNORMAL_THRESHOLD (default 0.25). The label is a
flag for expert review, not a diagnosis. Lowering the threshold flags more cells, which favours recall.
0.25 was chosen on the validation split of Leukemia-Cells v1 (best F1(abnormal) 0.892, recall 0.935);
re-tune it on the validation split whenever the weights change.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:   # imported as ai.classification (pipeline): no generic names on sys.path
    from .model import load_checkpoint
    from .transforms import eval_transform
else:             # imported by the scripts in this folder (evaluate.py)
    from model import load_checkpoint
    from transforms import eval_transform

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
ABNORMAL_THRESHOLD = float(os.environ.get("HACKCURE_ABNORMAL_THRESHOLD", "0.25"))


class Classifier:
    def __init__(self, weights: str | Path, device: str = "cpu", threshold: float = ABNORMAL_THRESHOLD):
        self.model, ckpt = load_checkpoint(weights, device)
        self.device = device
        self.threshold = threshold
        self.transform = eval_transform(ckpt["img_size"])
        self.abnormal_idx = next(i for i, lab in ckpt["idx_to_label"].items() if lab == "abnormal")

    @torch.no_grad()
    def predict_proba(self, crops: list[np.ndarray]) -> np.ndarray:
        """p(abnormal) for each RGB uint8 crop (H, W, 3), as one batch."""
        if not crops:
            return np.zeros(0)
        x = torch.stack([self.transform(Image.fromarray(np.ascontiguousarray(c)).convert("RGB")) for c in crops])
        probs = torch.softmax(self.model(x.to(self.device)), dim=1)[:, self.abnormal_idx]
        return probs.cpu().numpy()

    def to_classification(self, p_abnormal: float) -> dict:
        if p_abnormal >= self.threshold:
            return {"label": "abnormal", "score": round(float(p_abnormal), 4)}
        return {"label": "normal", "score": round(1.0 - float(p_abnormal), 4)}


_classifier: Classifier | None = None


def get_classifier() -> Classifier:
    global _classifier
    if _classifier is None:
        weights = os.environ.get("HACKCURE_CLASSIFIER_WEIGHTS", str(DEFAULT_WEIGHTS))
        if not Path(weights).is_file():
            raise FileNotFoundError(f"classifier weights not found: {weights} "
                                    "(set HACKCURE_CLASSIFIER_WEIGHTS)")
        _classifier = Classifier(weights, device=os.environ.get("HACKCURE_DEVICE", "cpu"))
    return _classifier


def classify_cell(crop: np.ndarray) -> dict:
    """M3 contract (contracts/interfaces.py): classify one RGB uint8 cell crop.

    Returns {"label": "normal" | "abnormal", "score": probability of that label in [0, 1]}.
    """
    clf = get_classifier()
    return clf.to_classification(clf.predict_proba([crop])[0])


def classify_cells(crops: list[np.ndarray]) -> list[dict]:
    """Batch version of classify_cell, same output per crop, much faster on CPU for hundreds of
    cells. Optional for M4: the contract function is classify_cell."""
    clf = get_classifier()
    return [clf.to_classification(p) for p in clf.predict_proba(crops)]
