"""Non-negative pairwise-logistic meta learner.

The production meta learner minimizes pointwise log loss.  This challenger
instead learns weights from every positive-negative score difference, which is
a smooth ranking surrogate closer to average precision.  A one-dimensional
Platt layer is fitted afterwards so that the existing ``half_f1`` decision rule
still receives calibrated probabilities.

This module is experimental and does not change production defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

PAIRWISE_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _penalty(w: np.ndarray, target: str) -> tuple[float, np.ndarray]:
    if target == "zero":
        return float(w @ w), 2.0 * w
    if target == "uniform":
        centered = w - w.mean()
        return float(centered @ centered), 2.0 * centered
    raise ValueError(f"unknown shrinkage target: {target}")


def _pair_differences(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    positive = Z[y == 1]
    negative = Z[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("pairwise loss requires both classes")
    return (positive[:, None, :] - negative[None, :, :]).reshape(
        -1, Z.shape[1])


def fit_pairwise_weights(Z: np.ndarray, y: np.ndarray, alpha: float,
                         target: str = "zero") -> np.ndarray:
    """Fit non-negative expert weights with all positive-negative pairs."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=int)
    differences = _pair_differences(Z, y)

    def loss_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        margin = differences @ w
        wrong_order_probability = expit(-margin)
        penalty, penalty_grad = _penalty(w, target)
        loss = (np.logaddexp(0.0, -margin).mean() + alpha * penalty)
        grad = (-(differences.T @ wrong_order_probability) / len(differences)
                + alpha * penalty_grad)
        return float(loss), grad

    initial = np.full(Z.shape[1], 1.0 / Z.shape[1])
    result = minimize(
        loss_grad, initial, jac=True, bounds=[(0.0, None)] * Z.shape[1],
        method="L-BFGS-B", options={"maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"pairwise meta optimization failed: {result.message}")
    return result.x


def fit_platt(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a non-negative scale and intercept by ordinary log loss."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    def loss_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scale, intercept = theta
        margin = scale * score + intercept
        probability = expit(margin)
        residual = (probability - y) / len(y)
        loss = np.logaddexp(0.0, margin).mean() - np.mean(y * margin)
        return float(loss), np.array([score @ residual, residual.sum()])

    result = minimize(
        loss_grad, np.array([1.0, 0.0]), jac=True,
        bounds=[(0.0, None), (None, None)], method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Platt calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def apply_pairwise_meta(Z: np.ndarray, weights: np.ndarray,
                        scale: float, intercept: float) -> np.ndarray:
    return expit(scale * (np.asarray(Z) @ weights) + intercept)


def fit_pairwise_meta(Z: np.ndarray, y: np.ndarray, alpha: float,
                      target: str = "zero") -> tuple[np.ndarray, float, float]:
    weights = fit_pairwise_weights(Z, y, alpha, target)
    scale, intercept = fit_platt(np.asarray(Z) @ weights, y)
    return weights, scale, intercept


def select_pairwise_alpha(Z: np.ndarray, y: np.ndarray, seed: int,
                          grid=PAIRWISE_ALPHA_GRID, n_splits: int = 5,
                          target: str = "zero") -> tuple[float, float]:
    """Select regularization by inner-CV AP without looking at outer labels."""
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed).split(Z, y)
    fold_indices = list(folds)
    best_alpha, best_ap = float(grid[0]), -np.inf
    for alpha in grid:
        score = np.zeros(len(y), dtype=float)
        for train_idx, valid_idx in fold_indices:
            weights = fit_pairwise_weights(
                Z[train_idx], y[train_idx], float(alpha), target)
            score[valid_idx] = Z[valid_idx] @ weights
        ap = float(average_precision_score(y, score))
        if ap > best_ap:
            best_alpha, best_ap = float(alpha), ap
    return best_alpha, best_ap
