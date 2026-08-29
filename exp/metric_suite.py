"""Binary classification metrics used by loss-function experiments.

The competition score is F1, but F1 alone cannot tell whether a change came
from ranking, calibration, or the operating threshold.  This module reports
those components together so that a loss is not accepted for moving a single
thresholded number by chance.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def calibration_intercept_slope(y: np.ndarray,
                                p: np.ndarray) -> tuple[float, float]:
    """Return post-hoc calibration intercept and slope for diagnostics only."""
    model = LogisticRegression(C=1e4, max_iter=2000, random_state=0)
    model.fit(_logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def quantile_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency expected calibration error.

    Equal-width bins are mostly empty for the current highly separated model.
    Quantile bins keep this descriptive diagnostic usable at n=260.  ECE is
    not used for model selection because it remains sample-size sensitive.
    """
    order = np.argsort(p)
    chunks = np.array_split(order, min(n_bins, len(order)))
    return float(sum(
        len(idx) / len(y) * abs(float(np.mean(y[idx])) - float(np.mean(p[idx])))
        for idx in chunks if len(idx)
    ))


def score_predictions(y: np.ndarray, p: np.ndarray,
                      threshold: float) -> dict[str, float | int]:
    """Score probabilities at one threshold, including ranking and calibration."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cal_intercept, cal_slope = calibration_intercept_slope(y, p)
    positive = y == 1
    negative = ~positive
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "positive_logloss": float(-np.log(p[positive]).mean()),
        "negative_logloss": float(-np.log1p(-p[negative]).mean()),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
        "ece_quantile": quantile_ece(y, p),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold": float(threshold),
        "pos_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
