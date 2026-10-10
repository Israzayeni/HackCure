# ai/classification — M3 (Sarah)

**Branch:** `ai/classification`

**Deliverable:** `classify_cell(crop) -> {"label": "normal" | "abnormal", "score": float}` (trained
ResNet18 or EfficientNet-B0), plus the accuracy and F1 on a held-out set.
Signature and docstring: [`contracts/interfaces.py`](../../contracts/interfaces.py).
Schema: [`contracts/classification.schema.json`](../../contracts/classification.schema.json).
"abnormal" is a flag for expert review, not a diagnosis.

## Files

| File | What |
|---|---|
| `transforms.py` | Preprocessing shared by training and inference; folder name → label map (`FOLDER_TO_LABEL`) |
| `model.py` | Builds ResNet18 / EfficientNet-B0 with a 2-class head; checkpoint save/load |
| `train.py` | Fine-tunes on an ImageFolder dataset; saves `best.pt` (best val F1 on abnormal) and `last.pt` every epoch |
| `evaluate.py` | Accuracy, F1, precision, recall, confusion matrix on a held-out split → `metrics.json` |
| `classifier.py` | `classify_cell(crop)` for the pipeline, plus the batch helper `classify_cells(crops)` |
| `tests/` | Contract tests with an untrained model: `python -m pytest ai/classification/tests` |

## Dataset

Roboflow **Leukemia-Cells** (version 1), classification export: `train/`, `valid/`, `test/`, each with
`effected-cells/` (→ abnormal) and `normal-cells/` (→ normal). TODO: link, license, image counts, and whether
Roboflow added augmented copies (see `README.roboflow.txt`). Stained smear images, so there is a domain gap with
unstained Malassez images; the training augmentations randomize or remove colour for that reason.

## Train (Colab, GPU runtime)

```python
from google.colab import drive; drive.mount('/content/drive')
# upload transforms.py, model.py, train.py, classifier.py, evaluate.py to /content/m3/ (or git clone the repo)
!pip -q install scikit-learn
!python /content/m3/train.py --data /content/Leukemia-Cells-1 --out /content/drive/MyDrive/hackcure
```

Choose the checkpoint on `valid` (done automatically), then report the final numbers **once** on `test`:

```python
!python /content/m3/evaluate.py --weights /content/drive/MyDrive/hackcure/best.pt \
    --data /content/Leukemia-Cells-1/test --out /content/drive/MyDrive/hackcure/metrics.json
```

Copy `best.pt` to `ai/classification/weights/best.pt` (gitignored, never committed) and `metrics.json` to
`ai/classification/metrics.json`.

## Use in the pipeline (M4)

```python
from ai.classification import classify_cell, classify_cells
classify_cell(crop)            # crop: RGB uint8 (H, W, 3)
```

Environment variables: `HACKCURE_CLASSIFIER_WEIGHTS` (default `ai/classification/weights/best.pt`),
`HACKCURE_ABNORMAL_THRESHOLD` (default 0.25, chosen on the validation split; lower flags more cells), `HACKCURE_DEVICE` (default `cpu`).
For hundreds of cells per image, `classify_cells` runs one batch and is much faster on CPU.

**Until H7:** develop against `fakes/crops` (`ImageFolder` layout, 120 drawn crops). Never train a real model
on them and never report metrics measured on them. **At integration:** `classify_cell` replaces
`fakes.stubs.fake_classify_cell`.
