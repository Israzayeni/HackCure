"""Evaluate a trained classifier on a held-out split and write metrics.json (M3 deliverable).

Run it once, on the test split, after choosing the checkpoint on the validation split:
    python ai/classification/evaluate.py --weights ai/classification/weights/best.pt \
        --data /content/Leukemia-Cells-1/test --out ai/classification/metrics.json

Choose the threshold on the validation split, never on test:
    python ai/classification/evaluate.py --weights .../best.pt --data /content/Leukemia-Cells-1/valid \
        --out /content/val_metrics.json --sweep

Predictions go through classifier.predict_proba with the same threshold as classify_cell,
so the reported numbers are those of the function the pipeline calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import ABNORMAL_THRESHOLD, Classifier  # noqa: E402
from metrics import classification_metrics  # noqa: E402
from transforms import FOLDER_TO_LABEL  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def print_sweep(y_true: np.ndarray, p_abn: np.ndarray) -> None:
    rows = [classification_metrics(y_true, p_abn, round(float(t), 2)) for t in np.arange(0.05, 0.96, 0.05)]
    print("\nthreshold  F1 macro  F1 normal  F1 abn  recall normal  recall abn")
    for m in rows:
        print(f"{m['threshold']:9.2f}  {m['f1_macro']:8.3f}  {m['f1_normal']:9.3f}  {m['f1_abnormal']:6.3f}"
              f"  {m['recall_normal']:13.3f}  {m['recall_abnormal']:10.3f}")
    best = max(rows, key=lambda m: m["f1_macro"])
    print(f"\nbest F1 macro at threshold {best['threshold']} ({best['f1_macro']:.3f}). "
          "Use it only if this split is the validation split: "
          f"HACKCURE_ABNORMAL_THRESHOLD={best['threshold']}, evaluate.py --threshold {best['threshold']}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="split folder: <data>/<class folder>/<images>")
    p.add_argument("--out", default="ai/classification/metrics.json")
    p.add_argument("--threshold", type=float, default=ABNORMAL_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sweep", action="store_true", help="also print the metrics for thresholds 0.05..0.95")
    args = p.parse_args()

    clf = Classifier(args.weights, threshold=args.threshold)
    paths, y_true = [], []
    for folder in sorted(Path(args.data).iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in FOLDER_TO_LABEL:
            raise SystemExit(f"Folder {folder.name!r} has no label in FOLDER_TO_LABEL")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                paths.append(f)
                y_true.append(FOLDER_TO_LABEL[folder.name] == "abnormal")

    p_abn = []
    for i in range(0, len(paths), args.batch_size):
        crops = [np.asarray(Image.open(f).convert("RGB")) for f in paths[i:i + args.batch_size]]
        p_abn.extend(clf.predict_proba(crops).tolist())
    y_true, p_abn = np.array(y_true), np.array(p_abn)

    if args.sweep:
        print_sweep(y_true, p_abn)
    metrics = {"weights": str(args.weights), "data": str(args.data),
               **classification_metrics(y_true, p_abn, args.threshold)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
