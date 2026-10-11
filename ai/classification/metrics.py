"""Classification metrics shared by train.py (validation, after every epoch) and evaluate.py (held-out split).

y_true is True for abnormal cells, p_abnormal is the predicted probability of abnormal. A cell is
predicted abnormal when p_abnormal >= threshold, the same rule as classify_cell. train.py uses 0.5,
which is the argmax of the two softmax outputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true, p_abnormal, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    p = np.asarray(p_abnormal, dtype=float)
    y_pred = p >= threshold
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[True, False], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    try:
        auc = round(float(roc_auc_score(y_true, p)), 4)
    except ValueError:   # only one class present
        auc = None

    def r(v):
        return round(float(v), 4)

    return {
        "n": int(len(y_true)),
        "n_abnormal": int(y_true.sum()),
        "threshold": threshold,
        "accuracy": r((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "f1_abnormal": r(f1[0]),
        "precision_abnormal": r(prec[0]),
        "recall_abnormal": r(rec[0]),
        "f1_normal": r(f1[1]),
        "precision_normal": r(prec[1]),
        "recall_normal": r(rec[1]),
        "f1_macro": r(f1.mean()),
        "roc_auc": auc,
        "confusion_matrix": {"rows_true_cols_pred": ["normal", "abnormal"], "values": cm.tolist()},
    }
