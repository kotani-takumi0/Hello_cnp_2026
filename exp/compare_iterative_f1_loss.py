"""Strict holdout test of the iterative F1 cost-surrogate meta loss.

The baseline, fixed positive-weight control, and iterative candidate share the
same expert predictions and outer split.  The candidate repeatedly estimates
F1 from honest meta OOF decisions and updates the weighted-logistic costs
before applying the same cross-fitted Platt + half-F1 deployment rule.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_loss_families import (  # noqa: E402
    ALPHA_GRID, EPS, _best_f1, _cross_fitted_margins, _fit_one_loss,
    _fit_platt, _inputs, _standardized_prediction,
)
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS  # noqa: E402
from holdout_check import _stack, fit_predict_experts  # noqa: E402
from meta_f1_iterative import select_iterative_alpha  # noqa: E402
from meta_loss_lab import (  # noqa: E402
    LOSS_SPECS, LossSpec, fit_loss_meta, predict_margin,
)
from metric_suite import score_predictions  # noqa: E402


BASELINE = "logloss"
FIXED_CONTROL = "weighted_logloss_pos15"
CANDIDATE = "iterative_f1_cost_logloss"


def _fit_iterative_candidate(
        Z_train: np.ndarray, y_train: np.ndarray, Z_holdout: np.ndarray,
        y_holdout: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    alpha, inner_ap, iterative = select_iterative_alpha(
        Z_train, y_train, folds, ALPHA_GRID, initial_f1=0.8)
    spec = LossSpec(
        CANDIDATE, "weighted_logloss",
        {"positive_weight": iterative.positive_weight})

    standardized_oof_margin = _cross_fitted_margins(
        Z_train, y_train, spec, alpha, folds)
    calibration_scale, calibration_intercept = _fit_platt(
        standardized_oof_margin, y_train)
    calibrated_oof = expit(
        calibration_scale * standardized_oof_margin + calibration_intercept)

    weights, intercept = fit_loss_meta(Z_train, y_train, spec, alpha)
    _, standardized_holdout_margin = _standardized_prediction(
        Z_train, Z_holdout, weights, intercept)
    calibrated_holdout = expit(
        calibration_scale * standardized_holdout_margin
        + calibration_intercept)

    train_best_f1, _ = _best_f1(y_train, calibrated_oof)
    threshold = train_best_f1 / 2.0
    outer_oracle_f1, outer_oracle_threshold = _best_f1(
        y_holdout, calibrated_holdout)
    calibrated_score = score_predictions(
        y_holdout, calibrated_holdout, threshold)

    raw_holdout_margin = predict_margin(Z_holdout, weights, intercept)
    raw_holdout_probability = expit(raw_holdout_margin)
    direct_score = score_predictions(
        y_holdout, raw_holdout_probability, 0.5)
    return {
        "loss": CANDIDATE,
        "loss_kind": "iterative_weighted_logloss",
        "penalty_target": "zero",
        "alpha": alpha,
        "inner_ap": inner_ap,
        "estimated_f1_cost": iterative.estimated_f1,
        "positive_weight": iterative.positive_weight,
        "iterations": iterative.iterations,
        "convergence_reason": iterative.convergence_reason,
        "calibration_scale": calibration_scale,
        "calibration_intercept_fitted": calibration_intercept,
        "raw_logloss": float(log_loss(
            y_holdout,
            np.clip(expit(standardized_holdout_margin), EPS, 1.0 - EPS))),
        "raw_brier": float(brier_score_loss(
            y_holdout, expit(standardized_holdout_margin))),
        "train_oof_f1": train_best_f1,
        "outer_oracle_f1": outer_oracle_f1,
        "outer_oracle_threshold": outer_oracle_threshold,
        "weight_l1": float(weights.sum()),
        "weight_nonzero": int((weights > 1e-8).sum()),
        "direct_f1": direct_score["f1"],
        "direct_precision": direct_score["precision"],
        "direct_recall": direct_score["recall"],
        "direct_pos_rate": direct_score["pos_rate"],
        **calibrated_score,
    }


def _one_repeat(X, y, text, org, e4, columns, seed: int,
                test_size: float) -> list[dict]:
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

    fixed_specs = {
        spec.name: spec for spec in LOSS_SPECS
        if spec.name in {BASELINE, FIXED_CONTROL}
    }
    rows = []
    for name in (BASELINE, FIXED_CONTROL):
        row = _fit_one_loss(
            Z_train, y_train, Z_holdout, y_holdout,
            fixed_specs[name], folds)
        row.update({
            "estimated_f1_cost": np.nan,
            "positive_weight": (
                1.0 if name == BASELINE else 1.5),
            "iterations": 1,
            "convergence_reason": "fixed_loss",
            "direct_f1": np.nan,
            "direct_precision": np.nan,
            "direct_recall": np.nan,
            "direct_pos_rate": np.nan,
        })
        rows.append({"seed": seed, **row})
    rows.append({"seed": seed, **_fit_iterative_candidate(
        Z_train, y_train, Z_holdout, y_holdout, folds)})
    return rows


def _report(frame: pd.DataFrame) -> None:
    metrics = [
        "auc", "ap", "logloss", "brier", "f1", "precision", "recall",
        "mcc", "outer_oracle_f1", "threshold", "pos_rate", "alpha",
        "positive_weight", "estimated_f1_cost", "iterations",
    ]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("loss")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
    iterative = frame[frame.loss == CANDIDATE]
    print("\n=== iterative direct decision before post-hoc calibration ===")
    print(iterative[[
        "direct_f1", "direct_precision", "direct_recall", "direct_pos_rate",
    ]].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))

    pivot = frame.pivot(index="seed", columns="loss")
    for reference in (BASELINE, FIXED_CONTROL):
        print(f"\n=== {CANDIDATE} - {reference}: paired deltas ===")
        for metric in ("f1", "ap", "logloss", "brier", "mcc",
                       "outer_oracle_f1"):
            delta = pivot[(metric, CANDIDATE)] - pivot[(metric, reference)]
            benefit = -delta if metric in {"logloss", "brier"} else delta
            print(
                f"{metric:16s} delta={delta.mean():+.5f}  "
                f"wins={int((benefit > 1e-12).sum())}/"
                f"ties={int((np.abs(benefit) <= 1e-12).sum())}/"
                f"losses={int((benefit < -1e-12).sum())}")


def run(n_reps: int, seed_offset: int, test_size: float,
        output: str) -> pd.DataFrame:
    X, y, text, org, e4, columns = _inputs()
    records = []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        records.extend(_one_repeat(
            X, y, text, org, e4, columns, seed, test_size))
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _report(frame)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=246)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_iterative_f1_seed246_5rep.csv")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output)
