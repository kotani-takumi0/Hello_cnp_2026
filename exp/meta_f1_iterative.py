"""Iterative F1 surrogate via cost-sensitive logistic meta learning.

F1 is a pseudo-linear ratio of false-negative and false-positive rates.  If a
current estimate of the attainable F1 is ``q``, the corresponding linearized
costs are

    false negative: 1 - q / 2
    false positive: q / 2

and their ratio is ``(2 - q) / q``.  We use weighted logistic loss as the
convex surrogate for that cost-sensitive subproblem, estimate a new q from
cross-fitted decisions, and repeat to a fixed point.  This is a Dinkelbach-like
outer loop around an ordinary non-negative logistic meta learner.

The module changes no production defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from meta_loss_lab import LossSpec, fit_loss_meta, predict_margin


F1_MIN = 0.05
F1_MAX = 0.99


@dataclass(frozen=True)
class IterationStep:
    iteration: int
    input_f1: float
    observed_f1: float
    positive_weight: float
    positive_predictions: int


@dataclass(frozen=True)
class IterativeF1Result:
    estimated_f1: float
    positive_weight: float
    raw_oof_margin: np.ndarray
    iterations: int
    converged: bool
    convergence_reason: str
    history: tuple[IterationStep, ...]


def positive_weight_from_f1(f1_estimate: float) -> float:
    """Return the FN/FP cost ratio implied by a current F1 estimate."""
    q = float(np.clip(f1_estimate, F1_MIN, F1_MAX))
    return (2.0 - q) / q


def weighted_logloss_spec(f1_estimate: float) -> LossSpec:
    weight = positive_weight_from_f1(f1_estimate)
    return LossSpec(
        name="iterative_f1_cost_logloss",
        kind="weighted_logloss",
        params={"positive_weight": weight},
    )


def cross_fitted_raw_margin(
        Z: np.ndarray, y: np.ndarray, alpha: float, f1_estimate: float,
        folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Fit the cost-sensitive subproblem and return honest OOF margins."""
    spec = weighted_logloss_spec(f1_estimate)
    margin = np.zeros(len(y), dtype=float)
    for train_index, valid_index in folds:
        weights, intercept = fit_loss_meta(
            Z[train_index], y[train_index], spec, alpha)
        margin[valid_index] = predict_margin(
            Z[valid_index], weights, intercept)
    return margin


def fit_iterative_f1(
        Z: np.ndarray, y: np.ndarray, alpha: float,
        folds: list[tuple[np.ndarray, np.ndarray]], initial_f1: float = 0.8,
        damping: float = 0.5, tolerance: float = 1e-5,
        max_iterations: int = 60) -> IterativeF1Result:
    """Alternate cost construction and cross-fitted F1 estimation."""
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    q = float(np.clip(initial_f1, F1_MIN, F1_MAX))
    history: list[IterationStep] = []
    label_history: list[np.ndarray] = []
    input_history: list[float] = []
    convergence_reason = "max_iterations_best"
    margin = np.zeros(len(y), dtype=float)
    best_margin = margin.copy()
    best_step: IterationStep | None = None

    for iteration in range(1, max_iterations + 1):
        weight = positive_weight_from_f1(q)
        margin = cross_fitted_raw_margin(Z, y, alpha, q, folds)
        label = (margin >= 0.0).astype(int)
        observed = float(f1_score(y, label))
        step = IterationStep(
            iteration=iteration,
            input_f1=q,
            observed_f1=observed,
            positive_weight=weight,
            positive_predictions=int(label.sum()),
        )
        history.append(step)
        residual = abs(observed - q)
        if (best_step is None
                or observed > best_step.observed_f1 + 1e-12
                or (abs(observed - best_step.observed_f1) <= 1e-12
                    and residual
                    < abs(best_step.observed_f1 - best_step.input_f1))):
            best_step = step
            best_margin = margin.copy()
        if abs(observed - q) <= tolerance:
            convergence_reason = "fixed_point"
            break
        if (len(label_history) >= 2
                and np.array_equal(label, label_history[-2])
                and abs(q - input_history[-2]) <= tolerance):
            convergence_reason = "two_cycle_best"
            break
        label_history.append(label.copy())
        input_history.append(q)
        q = float(np.clip(
            (1.0 - damping) * q + damping * observed, F1_MIN, F1_MAX))

    assert best_step is not None
    converged = convergence_reason != "max_iterations_best"
    if convergence_reason == "fixed_point":
        selected_step = history[-1]
        selected_margin = margin
    else:
        selected_step = best_step
        selected_margin = best_margin
    return IterativeF1Result(
        estimated_f1=selected_step.observed_f1,
        positive_weight=selected_step.positive_weight,
        raw_oof_margin=selected_margin,
        iterations=len(history),
        converged=converged,
        convergence_reason=convergence_reason,
        history=tuple(history),
    )


def select_iterative_alpha(
        Z: np.ndarray, y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]], alpha_grid,
        initial_f1: float = 0.8) -> tuple[float, float, IterativeF1Result]:
    """Select L2 strength by AP after each alpha reaches its F1 fixed point."""
    candidates = []
    for alpha in alpha_grid:
        result = fit_iterative_f1(
            Z, y, float(alpha), folds, initial_f1=initial_f1)
        ap = float(average_precision_score(y, result.raw_oof_margin))
        candidates.append((float(alpha), ap, result))
    stable = [item for item in candidates if item[2].converged]
    pool = stable if stable else candidates
    best_alpha, best_ap, best_result = max(pool, key=lambda item: item[1])
    return best_alpha, best_ap, best_result
