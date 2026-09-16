from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, probabilities))
        metrics["auprc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics
