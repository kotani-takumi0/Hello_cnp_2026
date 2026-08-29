"""Custom LightGBM training objectives for Expert-level F1-approximation tests.

The meta-blend layer already rejected four independent F1-approximation losses
(see documents/sigmoidf1_loss_audit_20260827.md and its predecessors). This
module applies the same two loss families -- an iterative cost-sensitive
weighted logloss (Parambath et al. 2014 reduction, Dinkelbach-like outer loop)
and sigmoidF1 (Benedict et al. 2021, arXiv:2108.10566) -- directly to a single
LightGBM expert's own training objective instead, following LightGBM's custom
fobj/feval convention (raw preds are logit-scale margins; the boosting session
must carry an explicit init_score since a custom objective bypasses LightGBM's
automatic boost-from-average handling).

sigmoidF1 is a whole-batch, non-separable statistic, so its exact Hessian has
nonzero cross terms between instances -- incompatible with LightGBM's diagonal
Newton step. We use the standard practice for listwise/aggregate losses in
gradient boosting: differentiate holding every other instance's raw score
fixed (the "diagonal" approximation), then floor the result to stay positive
since the loss surface is non-convex and an exact second derivative can go
negative.

This module changes no production defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

HESS_FLOOR = 1e-3


def base_score(y: np.ndarray) -> float:
    """Logit of the training fold's positive rate, used as init_score."""
    rate = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    return float(np.log(rate / (1.0 - rate)))


def weighted_logloss_fobj(positive_weight: float):
    def fobj(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label()
        p = expit(preds)
        weight = np.where(y == 1, positive_weight, 1.0)
        grad = weight * (p - y)
        hess = weight * p * (1.0 - p)
        return grad, hess
    return fobj


def weighted_logloss_feval(positive_weight: float):
    def feval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
        y = dataset.get_label()
        p = np.clip(expit(preds), 1e-9, 1.0 - 1e-9)
        weight = np.where(y == 1, positive_weight, 1.0)
        terms = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        value = float(np.average(terms, weights=weight))
        return "weighted_logloss", value, False
    return feval


def sigmoid_f1_fobj(beta: float, eta: float):
    """Diagonal-Hessian sigmoidF1 objective; preds are raw (logit) margins."""
    def fobj(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label().astype(float)
        preds = np.asarray(preds, dtype=float)
        s = expit(beta * (preds + eta))
        s_prime = beta * s * (1.0 - s)
        s_double_prime = beta * beta * s * (1.0 - s) * (1.0 - 2.0 * s)

        true_positive = float(y @ s)
        denominator = float(s.sum() + y.sum())
        denominator = max(denominator, 1e-9)

        d_tp = y * s_prime
        d_denominator = s_prime
        d_f1 = 2.0 * (d_tp * denominator - true_positive * d_denominator
                      ) / denominator ** 2
        grad = -d_f1

        dd_tp = y * s_double_prime
        dd_denominator = s_double_prime
        dd_f1 = (2.0 * (dd_tp * denominator
                        - true_positive * dd_denominator) / denominator ** 2
                 - 2.0 * d_denominator * d_f1 / denominator)
        hess = np.maximum(-dd_f1, HESS_FLOOR)
        return grad, hess
    return fobj


def sigmoid_f1_feval(beta: float, eta: float):
    def feval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
        y = dataset.get_label().astype(float)
        s = expit(beta * (np.asarray(preds, dtype=float) + eta))
        true_positive = float(y @ s)
        denominator = float(s.sum() + y.sum()) + 1e-9
        f1 = 2.0 * true_positive / denominator
        return "sigmoid_f1", f1, True
    return feval
