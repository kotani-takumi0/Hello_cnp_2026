"""Low-dimensional meta learners for controlled loss-function experiments.

The production ensemble combines a small number of expert logits.  This file
keeps that model class fixed and changes only the empirical loss.  Every loss
returns an analytic gradient with respect to the sample margins; the common
optimizer then applies the same non-negative weight constraint and L2
regularization.

This module is experimental.  Production defaults are intentionally untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


EPS = 1e-9


@dataclass(frozen=True)
class LossSpec:
    """Definition of one empirical loss and its regularization target."""

    name: str
    kind: str
    params: dict[str, float] = field(default_factory=dict)
    penalty_target: str = "zero"


LOSS_SPECS = (
    LossSpec("logloss", "logloss"),
    LossSpec("logloss_uniform", "logloss", penalty_target="uniform"),
    LossSpec("brier", "brier"),
    LossSpec("weighted_logloss_pos15", "weighted_logloss",
             {"positive_weight": 1.5}),
    LossSpec("focal_gamma1", "focal", {"gamma_positive": 1.0,
                                         "gamma_negative": 1.0}),
    LossSpec("focal_gamma2", "focal", {"gamma_positive": 2.0,
                                         "gamma_negative": 2.0}),
    LossSpec("asymmetric_focal_neg2", "focal",
             {"gamma_positive": 0.0, "gamma_negative": 2.0}),
    LossSpec("logloss_softf1_025", "logloss_softf1", {"mix": 0.25}),
    LossSpec("logloss_pairwise_025", "logloss_pairwise", {"mix": 0.25}),
    LossSpec("sigmoid_f1_b1e0", "sigmoid_f1", {"beta": 1.0, "eta": 0.0}),
    LossSpec("sigmoid_f1_b7e1", "sigmoid_f1", {"beta": 7.0, "eta": 1.0}),
    LossSpec("sigmoid_f1_b20e15", "sigmoid_f1", {"beta": 20.0, "eta": 1.5}),
)


def _logloss_margin(scores: np.ndarray,
                    y: np.ndarray) -> tuple[float, np.ndarray]:
    probability = expit(scores)
    loss = np.mean(np.logaddexp(0.0, scores) - y * scores)
    gradient = (probability - y) / len(y)
    return float(loss), gradient


def _weighted_logloss_margin(
        scores: np.ndarray, y: np.ndarray,
        positive_weight: float) -> tuple[float, np.ndarray]:
    probability = expit(scores)
    weights = np.where(y == 1, positive_weight, 1.0)
    normalizer = float(weights.sum())
    terms = np.logaddexp(0.0, scores) - y * scores
    loss = float(weights @ terms / normalizer)
    gradient = weights * (probability - y) / normalizer
    return loss, gradient


def _brier_margin(scores: np.ndarray,
                  y: np.ndarray) -> tuple[float, np.ndarray]:
    probability = expit(scores)
    residual = probability - y
    loss = float(np.mean(residual ** 2))
    gradient = 2.0 * residual * probability * (1.0 - probability) / len(y)
    return loss, gradient


def _focal_margin(scores: np.ndarray, y: np.ndarray,
                  gamma_positive: float,
                  gamma_negative: float) -> tuple[float, np.ndarray]:
    """Binary focal loss with a stable analytic margin derivative.

    ``gamma_positive`` and ``gamma_negative`` control only difficulty
    focusing; no implicit class-frequency weight is applied.  Gamma zero is
    exactly ordinary log loss.
    """
    signed_label = 2.0 * y - 1.0
    correct_probability = np.clip(expit(signed_label * scores), EPS, 1.0 - EPS)
    gamma = np.where(y == 1, gamma_positive, gamma_negative)
    modulation = (1.0 - correct_probability) ** gamma
    loss = -float(np.mean(modulation * np.log(correct_probability)))
    gradient = (
        signed_label * modulation
        * (gamma * correct_probability * np.log(correct_probability)
           - (1.0 - correct_probability))
        / len(y)
    )
    return loss, gradient


def _soft_f1_margin(scores: np.ndarray,
                    y: np.ndarray) -> tuple[float, np.ndarray]:
    probability = expit(scores)
    true_positive = float(y @ probability)
    denominator = float(probability.sum() + y.sum()) + EPS
    soft_f1 = 2.0 * true_positive / denominator
    derivative_probability = -2.0 * (
        y * denominator - true_positive) / denominator ** 2
    gradient = derivative_probability * probability * (1.0 - probability)
    return float(1.0 - soft_f1), gradient


def _sigmoid_f1_margin(scores: np.ndarray, y: np.ndarray, beta: float,
                       eta: float) -> tuple[float, np.ndarray]:
    """sigmoidF1 (Benedict et al. 2021, arXiv:2108.10566), used standalone.

    ``S(u; beta, eta) = sigmoid(beta * (u + eta))`` replaces the hard
    threshold step used to build soft TP/FP/FN counts.  ``beta=1, eta=0``
    reduces exactly to the plain soft-F1 relaxation (``_soft_f1_margin``);
    the extra slope/offset let the surrogate sit closer to the true
    indicator than the naive version the paper reports as underperforming.
    """
    surrogate = expit(beta * (scores + eta))
    true_positive = float(y @ surrogate)
    denominator = float(surrogate.sum() + y.sum()) + EPS
    soft_f1 = 2.0 * true_positive / denominator
    derivative_surrogate = -2.0 * (
        y * denominator - true_positive) / denominator ** 2
    gradient = (derivative_surrogate * beta
                * surrogate * (1.0 - surrogate))
    return float(1.0 - soft_f1), gradient


def _pairwise_margin(scores: np.ndarray,
                     y: np.ndarray) -> tuple[float, np.ndarray]:
    positive_index = np.flatnonzero(y == 1)
    negative_index = np.flatnonzero(y == 0)
    if not len(positive_index) or not len(negative_index):
        raise ValueError("pairwise loss requires both classes")
    margin = (scores[positive_index, None]
              - scores[None, negative_index])
    wrong_order_probability = expit(-margin)
    pair_count = margin.size
    loss = float(np.logaddexp(0.0, -margin).mean())
    gradient = np.zeros_like(scores)
    gradient[positive_index] = -wrong_order_probability.sum(axis=1) / pair_count
    gradient[negative_index] = wrong_order_probability.sum(axis=0) / pair_count
    return loss, gradient


def loss_margin(scores: np.ndarray, y: np.ndarray,
                spec: LossSpec) -> tuple[float, np.ndarray]:
    """Return empirical loss and derivative with respect to ``scores``."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if spec.kind == "logloss":
        return _logloss_margin(scores, y)
    if spec.kind == "weighted_logloss":
        return _weighted_logloss_margin(
            scores, y, spec.params["positive_weight"])
    if spec.kind == "brier":
        return _brier_margin(scores, y)
    if spec.kind == "focal":
        return _focal_margin(
            scores, y, spec.params["gamma_positive"],
            spec.params["gamma_negative"])
    if spec.kind == "logloss_softf1":
        mix = spec.params["mix"]
        base_loss, base_gradient = _logloss_margin(scores, y)
        decision_loss, decision_gradient = _soft_f1_margin(scores, y)
        return (
            float((1.0 - mix) * base_loss + mix * decision_loss),
            (1.0 - mix) * base_gradient + mix * decision_gradient,
        )
    if spec.kind == "sigmoid_f1":
        return _sigmoid_f1_margin(
            scores, y, spec.params["beta"], spec.params["eta"])
    if spec.kind == "logloss_pairwise":
        mix = spec.params["mix"]
        base_loss, base_gradient = _logloss_margin(scores, y)
        rank_loss, rank_gradient = _pairwise_margin(scores, y)
        return (
            float((1.0 - mix) * base_loss + mix * rank_loss),
            (1.0 - mix) * base_gradient + mix * rank_gradient,
        )
    raise ValueError(f"unknown loss kind: {spec.kind}")


def _penalty(weights: np.ndarray,
             target: str) -> tuple[float, np.ndarray]:
    if target == "zero":
        return float(weights @ weights), 2.0 * weights
    if target == "uniform":
        centered = weights - weights.mean()
        return float(centered @ centered), 2.0 * centered
    raise ValueError(f"unknown penalty target: {target}")


def fit_loss_meta(Z: np.ndarray, y: np.ndarray, spec: LossSpec,
                  alpha: float) -> tuple[np.ndarray, float]:
    """Fit non-negative meta weights and a free intercept."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=int)
    feature_count = Z.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = theta[:-1], theta[-1]
        scores = Z @ weights + intercept
        empirical_loss, score_gradient = loss_margin(scores, y, spec)
        penalty, penalty_gradient = _penalty(weights, spec.penalty_target)
        loss = empirical_loss + alpha * penalty
        gradient = np.concatenate([
            Z.T @ score_gradient + alpha * penalty_gradient,
            [score_gradient.sum()],
        ])
        return float(loss), gradient

    initial = np.concatenate([
        np.full(feature_count, 1.0 / feature_count), [0.0]
    ])
    result = minimize(
        objective, initial, jac=True,
        bounds=[(0.0, None)] * feature_count + [(None, None)],
        method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(
            f"{spec.name} optimization failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def predict_margin(Z: np.ndarray, weights: np.ndarray,
                   intercept: float) -> np.ndarray:
    return np.asarray(Z, dtype=float) @ weights + intercept

