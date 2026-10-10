"""Evaluate a trained classifier on a held-out split and write metrics.json (M3 deliverable).

Run it once, on the test split, after choosing the checkpoint on the validation split:
    python ai/classification/evaluate.py --weights ai/classification/weights/best.pt \
        --data /content/Leukemia-Cells-1/test --out ai/classification/metrics.json

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import ABNORMAL_THRESHOLD, Classifier  # noqa: E402
from transforms import FOLDER_TO_LABEL  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="split folder: <data>/<class folder>/<images>")
    p.add_argument("--out", default="ai/classification/metrics.json")
    p.add_argument("--threshold", type=float, default=ABNORMAL_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=64)
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
    y_true = np.array(y_true)
    y_pred = np.array(p_abn) >= args.threshold

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    metrics = {
        "weights": str(args.weights),
        "data": str(args.data),
        "n": len(paths),
        "n_abnormal": int(y_true.sum()),
        "threshold": args.threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_abnormal": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision_abnormal": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_abnormal": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": {"rows_true_cols_pred": ["normal", "abnormal"], "values": cm.tolist()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
