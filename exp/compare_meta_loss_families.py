"""Strict paired comparison of meta-level empirical loss families.

Only the meta objective changes.  Expert predictions, outer splits, feature
sets, regularization search grid, cross-fitted calibration, and the F1
threshold rule are shared.  The untouched outer holdout therefore measures the
whole decision procedure rather than an in-sample optimum.

Screening example:
  python exp/compare_meta_loss_families.py --n-reps 5 --seed-offset 226
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import (  # noqa: E402
    CURRENT_LINEAR_C, EXPERTS, _inputs,
)
from holdout_check import _stack, fit_predict_experts  # noqa: E402
from meta_loss_lab import (  # noqa: E402
    LOSS_SPECS, LossSpec, fit_loss_meta, predict_margin,
)
from metric_suite import score_predictions  # noqa: E402


ALPHA_GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
              1e-2, 3e-2, 1e-1, 3e-1, 1.0)
EPS = 1e-6


def _fit_platt(score: np.ndarray,
               y: np.ndarray) -> tuple[float, float]:
    """Unregularized two-parameter Platt fit on cross-fitted margins."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scale, intercept = theta
        margin = scale * score + intercept
        probability = expit(margin)
        residual = (probability - y) / len(y)
        loss = np.mean(np.logaddexp(0.0, margin) - y * margin)
        return float(loss), np.array([score @ residual, residual.sum()])

    result = minimize(
        objective, np.array([1.0, 0.0]), jac=True,
        bounds=[(0.0, None), (None, None)], method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Platt optimization failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _standardized_prediction(
        Z_fit: np.ndarray, Z_target: np.ndarray, weights: np.ndarray,
        intercept: float) -> tuple[np.ndarray, np.ndarray]:
    """Put fold and full-fit margins on a comparable location/scale."""
    fit_score = predict_margin(Z_fit, weights, intercept)
    target_score = predict_margin(Z_target, weights, intercept)
    center = float(fit_score.mean())
    scale = float(fit_score.std())
    if scale < 1e-8:
        scale = 1.0
    return ((fit_score - center) / scale,
            (target_score - center) / scale)


def _select_alpha(Z: np.ndarray, y: np.ndarray, spec: LossSpec,
                  folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Select regularization on A only, using AP as a stable rank criterion."""
    best_alpha, best_ap = float(ALPHA_GRID[0]), -np.inf
    for alpha in ALPHA_GRID:
        prediction = np.zeros(len(y), dtype=float)
        for train_index, valid_index in folds:
            weights, intercept = fit_loss_meta(
                Z[train_index], y[train_index], spec, float(alpha))
            _, prediction[valid_index] = _standardized_prediction(
                Z[train_index], Z[valid_index], weights, intercept)
        ap = float(average_precision_score(y, prediction))
        if ap > best_ap:
            best_alpha, best_ap = float(alpha), ap
    return best_alpha, best_ap


def _cross_fitted_margins(
        Z: np.ndarray, y: np.ndarray, spec: LossSpec, alpha: float,
        folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    margin = np.zeros(len(y), dtype=float)
    for train_index, valid_index in folds:
        weights, intercept = fit_loss_meta(
            Z[train_index], y[train_index], spec, alpha)
        _, margin[valid_index] = _standardized_prediction(
            Z[train_index], Z[valid_index], weights, intercept)
    return margin


def _best_f1(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate([
        [0.0], np.asarray(probability, dtype=float), [1.0]
    ]))
    values = np.asarray([
        f1_score(y, probability >= threshold) for threshold in thresholds
    ])
    index = int(np.argmax(values))
    return float(values[index]), float(thresholds[index])


def _fit_one_loss(
        Z_train: np.ndarray, y_train: np.ndarray, Z_holdout: np.ndarray,
        y_holdout: np.ndarray, spec: LossSpec,
        folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    alpha, inner_ap = _select_alpha(Z_train, y_train, spec, folds)
    oof_margin = _cross_fitted_margins(
        Z_train, y_train, spec, alpha, folds)
    calibration_scale, calibration_intercept = _fit_platt(
        oof_margin, y_train)
    oof_probability = expit(
        calibration_scale * oof_margin + calibration_intercept)

    weights, intercept = fit_loss_meta(Z_train, y_train, spec, alpha)
    full_train_margin, holdout_margin = _standardized_prediction(
        Z_train, Z_holdout, weights, intercept)
    raw_holdout_probability = expit(holdout_margin)
    holdout_probability = expit(
        calibration_scale * holdout_margin + calibration_intercept)

    oof_best_f1, _ = _best_f1(y_train, oof_probability)
    threshold = oof_best_f1 / 2.0
    outer_oracle_f1, outer_oracle_threshold = _best_f1(
        y_holdout, holdout_probability)
    score = score_predictions(y_holdout, holdout_probability, threshold)
    return {
        "loss": spec.name,
        "loss_kind": spec.kind,
        "penalty_target": spec.penalty_target,
        "alpha": alpha,
        "inner_ap": inner_ap,
        "calibration_scale": calibration_scale,
        "calibration_intercept_fitted": calibration_intercept,
        "raw_logloss": float(log_loss(
            y_holdout, np.clip(raw_holdout_probability, EPS, 1.0 - EPS))),
        "raw_brier": float(brier_score_loss(
            y_holdout, raw_holdout_probability)),
        "train_oof_f1": oof_best_f1,
        "outer_oracle_f1": outer_oracle_f1,
        "outer_oracle_threshold": outer_oracle_threshold,
        "weight_l1": float(weights.sum()),
        "weight_nonzero": int((weights > 1e-8).sum()),
        **score,
    }


def _one_repeat(X, y, text, org, e4, columns, seed: int,
                test_size: float, specs: tuple[LossSpec, ...]) -> list[dict]:
    all_indices = np.arange(len(y))
    train_index, holdout_index = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_index], y[holdout_index]
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(train_index, y_train))

    expert_oof = {name: np.zeros(len(train_index)) for name in EXPERTS}
    for inner_train, inner_valid in folds:
        predictions = fit_predict_experts(
            X, y, text, org, train_index[inner_train], train_index[inner_valid],
            seed, columns, e4=e4, e7_model="lr",
            linear_c=CURRENT_LINEAR_C, strict_lgbm=True,
        )
        for name in EXPERTS:
            expert_oof[name][inner_valid] = predictions[name]
    expert_holdout = fit_predict_experts(
        X, y, text, org, train_index, holdout_index, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True,
    )
    Z_train = _stack(expert_oof, EXPERTS)
    Z_holdout = _stack(expert_holdout, EXPERTS)

    rows = []
    for spec in specs:
        row = _fit_one_loss(
            Z_train, y_train, Z_holdout, y_holdout, spec, folds)
        rows.append({"seed": seed, **row})
    return rows


def _report(frame: pd.DataFrame) -> None:
    metrics = [
        "auc", "ap", "logloss", "brier", "f1", "precision", "recall",
        "mcc", "calibration_slope", "ece_quantile", "threshold",
        "pos_rate", "outer_oracle_f1", "raw_logloss", "raw_brier",
        "alpha", "weight_l1", "weight_nonzero",
    ]
    summary = frame.groupby("loss")[metrics].agg(["mean", "std"])
    order = (frame.groupby("loss")["f1"].mean()
             .sort_values(ascending=False).index)
    print("\n=== strict outer-holdout mean +/- std (ordered by F1) ===")
    print(summary.loc[order].T.to_string(float_format=lambda x: f"{x:.5f}"))

    pivot = frame.pivot(index="seed", columns="loss")
    baseline = "logloss"
    print("\n=== paired deltas versus logloss ===")
    for candidate in order:
        if candidate == baseline:
            continue
        print(f"\n[{candidate}]")
        for metric in ("f1", "ap", "logloss", "brier", "mcc",
                       "outer_oracle_f1"):
            delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
            direction = delta if metric not in ("logloss", "brier") else -delta
            print(
                f"{metric:16s} delta={delta.mean():+.5f}  "
                f"wins={int((direction > 1e-12).sum())}/"
                f"ties={int((np.abs(direction) <= 1e-12).sum())}/"
                f"losses={int((direction < -1e-12).sum())}")


def run(n_reps: int, seed_offset: int, test_size: float, output: str,
        selected_losses: list[str] | None = None) -> pd.DataFrame:
    specs = LOSS_SPECS
    if selected_losses:
        wanted = set(selected_losses)
        specs = tuple(spec for spec in specs if spec.name in wanted)
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"unknown losses: {sorted(missing)}")
    X, y, text, org, e4, columns = _inputs()
    records = []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        records.extend(_one_repeat(
            X, y, text, org, e4, columns, seed, test_size, specs))
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _report(frame)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=226)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_meta_loss_families_seed226_5rep.csv")
    parser.add_argument("--losses", nargs="*")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output, arguments.losses)
