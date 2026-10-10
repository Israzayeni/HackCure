# ai/classification — M3 (Sarah)

**Branch:** `ai/classification`

**Deliverable:** `classify_cell(crop) -> {"label": "normal" | "abnormal", "score": float}` (trained
ResNet18 or EfficientNet-B0), plus the accuracy and F1 on a held-out set.
Signature and docstring: [`contracts/interfaces.py`](../../contracts/interfaces.py).
Schema: [`contracts/classification.schema.json`](../../contracts/classification.schema.json).

**Until H7:** make crops from `fakes/images/<name>.png` using the boxes in `fakes/json/<name>.detections.json`
(`crop = image[y:y+h, x:x+w]`). The expected labels are in `fakes/json/<name>.classifications.json`, in the
same order as the detections. In the synthetic images, abnormal cells are darker and elongated.

**Ready-made crops:** `ImageFolder("fakes/crops")` gives 120 labelled 224×224 crops for testing your data/training/eval code. Do not train on them for real or report metrics on them. Details and open questions: root README, "Fake data for M3". Shared crop helper: `ai/common/crop.py`.

**At integration:** `classify_cell` replaces `fakes.stubs.fake_classify_cell`.
