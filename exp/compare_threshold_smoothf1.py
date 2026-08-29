"""Strict holdout test of a smoothed-F1 threshold rule against production `half_f1`.

Every loss-function approximation of F1 (meta layer and Expert/LightGBM layer,
see documents/sigmoidf1_loss_audit_20260827.md and
documents/expert_level_f1_loss_audit_20260827.md) was rejected today. This
script moves the same "approximate F1 with a smooth surrogate" idea to a
different stage of the pipeline: not the training loss, but the post-hoc
threshold selection on an already-trained, already-calibrated probability.

Production picks the single argmax of the empirical (jagged, tie-prone) F1(t)
curve via `train_best_f1 / 2` (the `half_f1` plug-in rule, itself a low-
variance simplification already validated over `oof_best`; see
holdout_check.py). `smooth_f1_k*` instead relaxes the indicator
1[p>=t] with a logistic kernel of sharpness `k` -- exactly the sigmoidF1
surrogate already implemented for training, but applied only at threshold-
selection time on frozen calibrated probabilities -- and takes the continuous
argmax of that smoothed curve via bounded scalar optimization. A smaller `k`
trades bias (the smoothed curve's peak drifts from the true discrete argmax)
for variance (less sensitivity to which few borderline points happen to land
near the cut); `k` values below span very smooth to near-discrete.

Everything else (Expert predictions, outer split, inner folds, meta loss,
alpha grid, cross-fitted Platt calibration) is shared with the current
production configuration and with today's other audits.

Screening example:
  python exp/compare_threshold_smoothf1.py --n-reps 5 --seed-offset 261
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_loss_families import (  # noqa: E402
    ALPHA_GRID, _best_f1, _cross_fitted_margins, _fit_platt,
    _select_alpha, _standardized_prediction,
)
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from holdout_check import (  # noqa: E402
    _fold_curve_thresholds, _plateau_midpoint, _stack, fit_predict_experts,
)
from meta_loss_lab import LOSS_SPECS  # noqa: E402
from metric_suite import score_predictions  # noqa: E402


LOGLOSS_SPEC = next(spec for spec in LOSS_SPECS if spec.name == "logloss")
SMOOTH_K_CANDIDATES = (15.0, 40.0, 100.0)


def _smooth_f1_threshold(pA: np.ndarray, yA: np.ndarray, k: float) -> float:
    """Continuous argmax of the sigmoid-relaxed soft-F1(t) curve.

    ``S(p; t, k) = sigmoid(k * (p - t))`` replaces ``1[p >= t]``; this is the
    same sigmoidF1 surrogate already implemented for training
    (``exp/lgbm_f1_objectives.py``, ``exp/meta_loss_lab.py``), applied here to
    a fixed frozen score instead of a trainable one.
    """
    total_pos = float(yA.sum())

    def negative_soft_f1(t: float) -> float:
        s = expit(k * (pA - t))
        true_positive = float(yA @ s)
        denominator = float(s.sum() + total_pos) + 1e-9
        return -(2.0 * true_positive / denominator)

    result = minimize_scalar(
        negative_soft_f1, bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-4})
    return float(result.x)


def _thresholds_for_repeat(
        oof_probability: np.ndarray, y_train: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    best_f1, best_threshold = _best_f1(y_train, oof_probability)
    out = {"half_f1": best_f1 / 2.0, "oof_best": best_threshold}
    fold_curves, _, _ = _fold_curve_thresholds(
        oof_probability, y_train, folds)
    out["plateau_mid"] = _plateau_midpoint(fold_curves)
    for k in SMOOTH_K_CANDIDATES:
        out[f"smooth_f1_k{int(k)}"] = _smooth_f1_threshold(
            oof_probability, y_train, k)
    return out


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

    alpha, _ = _select_alpha(Z_train, y_train, LOGLOSS_SPEC, folds)
    oof_margin = _cross_fitted_margins(Z_train, y_train, LOGLOSS_SPEC, alpha, folds)
    calibration_scale, calibration_intercept = _fit_platt(oof_margin, y_train)
    oof_probability = expit(calibration_scale * oof_margin + calibration_intercept)

    from meta_loss_lab import fit_loss_meta  # noqa: E402 (local import, cheap)
    weights, intercept = fit_loss_meta(Z_train, y_train, LOGLOSS_SPEC, alpha)
    _, holdout_margin = _standardized_prediction(
        Z_train, Z_holdout, weights, intercept)
    holdout_probability = expit(
        calibration_scale * holdout_margin + calibration_intercept)

    thresholds = _thresholds_for_repeat(oof_probability, y_train, folds)
    rows = []
    for rule, threshold in thresholds.items():
        score = score_predictions(y_holdout, holdout_probability, threshold)
        rows.append({"seed": seed, "rule": rule, "threshold": threshold, **score})
    return rows


def _report(frame: pd.DataFrame) -> None:
    metrics = ["f1", "precision", "recall", "mcc", "pos_rate", "threshold"]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("rule")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda v: f"{v:.5f}"))

    pivot = frame.pivot(index="seed", columns="rule")
    baseline = "half_f1"
    for rule in sorted(frame["rule"].unique()):
        if rule == baseline:
            continue
        print(f"\n=== {rule} - {baseline}: paired deltas ===")
        for metric in ("f1", "precision", "recall", "mcc"):
            delta = pivot[(metric, rule)] - pivot[(metric, baseline)]
            print(
                f"{metric:12s} delta={delta.mean():+.5f}  "
                f"wins={int((delta > 1e-12).sum())}/"
                f"ties={int((np.abs(delta) <= 1e-12).sum())}/"
                f"losses={int((delta < -1e-12).sum())}")


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
    parser.add_argument("--seed-offset", type=int, default=261)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_threshold_smoothf1_seed261_5rep.csv")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output)
