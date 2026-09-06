"""Classification metrics with honest intervals.

Accuracy intervals use Wilson and Clopper-Pearson because a bootstrap interval
degenerates to [1, 1] when a predictor makes zero errors. Bootstrap intervals are
reported only when there is something to resample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def confusion(y: Sequence[int], p: Sequence[int]) -> list[list[int]]:
    """``[[TN, FP], [FN, TP]]`` (sklearn order)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    return [[tn, fp], [fn, tp]]


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: Sequence[int],
    p: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> list[float]:
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        values.append(metric(yy, pp))
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def classification_metrics(
    y: Sequence[int],
    p: Sequence[int],
    prob: Sequence[float] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Accuracy / precision / recall / F1 / confusion with intervals, optional AUROC."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    n = len(y)
    correct = int((y == p).sum())
    out = {
        "n": n,
        "pos_rate": round(float(y.mean()), 4),
        "acc": round(float(accuracy_score(y, p)), 4),
        "acc_ci_wilson": [round(v, 4) for v in wilson_ci(correct, n)],
        "acc_ci_clopper_pearson": [round(v, 4) for v in clopper_pearson_ci(correct, n)],
        "precision": round(float(precision_score(y, p, zero_division=0)), 4),
        "recall": round(float(recall_score(y, p, zero_division=0)), 4),
        "f1": round(float(f1_score(y, p, zero_division=0)), 4),
        "confusion": confusion(y, p),
        "errors": int((y != p).sum()),
    }
    if out["errors"] == 0:
        out["f1_ci_bootstrap"] = "degenerate (zero errors)"
    else:
        out["f1_ci_bootstrap"] = [
            round(v, 4) for v in bootstrap_ci(lambda a, b: f1_score(a, b, zero_division=0), y, p, n_boot, seed)
        ]
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        out["auroc"] = round(float(roc_auc_score(y, prob)), 4)
        if len(np.unique(prob)) <= 2:
            out["auroc_note"] = (
                "scores are effectively hard labels; AUROC restates accuracy and adds no calibration or uncertainty"
            )
        out["auroc_ci_bootstrap"] = [round(v, 4) for v in bootstrap_ci(roc_auc_score, y, prob, n_boot, seed)]
    return out
