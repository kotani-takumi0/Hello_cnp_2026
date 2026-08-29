"""H64: strict holdout comparison of pointwise and pairwise meta losses.

Only the meta learner changes.  The eight expert predictions are trained once
per repeat and shared by both losses, giving an exact paired comparison while
keeping holdout B out of expert fitting, alpha selection, calibration, and the
threshold rule.

Example:
  python exp/compare_meta_losses.py --n-reps 5 --seed-offset 206
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import build_features, e4_inputs, expert_names  # noqa: E402
from expert_groups import (  # noqa: E402
    FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols,
    e1_finance_cols, e2_survey_cols, e6_manual_cols, e7_cross_cols,
    route_h2_anchor_cols,
)
from holdout_check import (  # noqa: E402
    _stack, _thresholds, fit_predict_experts,
)
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from meta_pairwise import (  # noqa: E402
    apply_pairwise_meta, fit_pairwise_meta, fit_platt, select_pairwise_alpha,
)
from metric_suite import score_predictions  # noqa: E402
from organization_features import load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402

EXPERTS = expert_names(cross=True, linear=True)
CURRENT_LINEAR_C = 0.03
BASELINE = "pointwise_logloss"
CALIBRATED_CONTROL = "pointwise_logloss_platt"
CANDIDATE = "pairwise_logistic_platt"


def _inputs():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, _ = build_features(
        train, test, cross=True, llm_makebuy=True,
        h2_restore=True, h7_restore=True,
    )
    text = train[TEXT_COL].fillna("").astype(str).values
    org = load_org_text(train)
    e4_input, _, e4_model = e4_inputs(
        train, train, org_embed=False, concat_embed=True)
    e4 = (e4_input, e4_model)

    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    core4 = {
        "LLM_need_score", "LLM_make_limitation_score",
        "LLM_buy_openness_score", "LLM_internal_training_capability_score",
    }
    c0 = [
        c for c in e0_anchor_cols(X.columns)
        if not c.startswith("DX展望_") and c not in missing
        and (not c.startswith("LLM_") or c in core4)
    ]
    c0b = [c for c in e0_anchor_cols(X.columns) if not c.startswith("LLM_")]
    c0, c0b = route_h2_anchor_cols(c0, c0b, "h2h7_both4")
    columns = (
        c0, c0b, e1_finance_cols(), e2_survey_cols(),
        e6_manual_cols(X.columns), e7_cross_cols(),
    )
    return X, y, text, org, e4, columns


def _one_repeat(X, y, text, org, e4, columns, seed: int,
                test_size: float) -> tuple[list[dict], dict]:
    all_indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train = y[train_idx]
    y_holdout = y[holdout_idx]

    expert_oof = {name: np.zeros(len(train_idx)) for name in EXPERTS}
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed).split(train_idx, y_train))
    for inner_train, inner_valid in folds:
        predictions = fit_predict_experts(
            X, y, text, org, train_idx[inner_train], train_idx[inner_valid],
            seed, columns, e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
            strict_lgbm=True,
        )
        for name in EXPERTS:
            expert_oof[name][inner_valid] = predictions[name]

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True,
    )
    Z_train = _stack(expert_oof, EXPERTS)
    Z_holdout = _stack(expert_holdout, EXPERTS)

    point_alpha, point_inner_ap = select_alpha(Z_train, y_train, seed=seed)
    point_weights, point_intercept = fit_meta(
        Z_train, y_train, alpha=point_alpha)
    point_train = apply_meta(Z_train, point_weights, point_intercept)
    point_holdout = apply_meta(Z_holdout, point_weights, point_intercept)
    point_scale, point_calibration_intercept = fit_platt(
        Z_train @ point_weights + point_intercept, y_train)
    point_platt_train = expit(
        point_scale * (Z_train @ point_weights + point_intercept)
        + point_calibration_intercept)
    point_platt_holdout = expit(
        point_scale * (Z_holdout @ point_weights + point_intercept)
        + point_calibration_intercept)

    pair_alpha, pair_inner_ap = select_pairwise_alpha(
        Z_train, y_train, seed=seed)
    pair_weights, pair_scale, pair_intercept = fit_pairwise_meta(
        Z_train, y_train, alpha=pair_alpha)
    pair_train = apply_pairwise_meta(
        Z_train, pair_weights, pair_scale, pair_intercept)
    pair_holdout = apply_pairwise_meta(
        Z_holdout, pair_weights, pair_scale, pair_intercept)

    rows = []
    for (loss, train_probability, holdout_probability, alpha, inner_ap,
         calibration_scale, calibration_offset) in (
        (BASELINE, point_train, point_holdout, point_alpha, point_inner_ap,
         1.0, 0.0),
        (CALIBRATED_CONTROL, point_platt_train, point_platt_holdout,
         point_alpha, point_inner_ap, point_scale,
         point_calibration_intercept),
        (CANDIDATE, pair_train, pair_holdout, pair_alpha, pair_inner_ap,
         pair_scale, pair_intercept),
    ):
        threshold = _thresholds(
            train_probability, y_train, holdout_probability, folds,
            ("half_f1",))["half_f1"]
        row = score_predictions(y_holdout, holdout_probability, threshold)
        rows.append({
            "seed": seed, "loss": loss, "alpha": alpha,
            "inner_ap": inner_ap,
            "fitted_calibration_scale": calibration_scale,
            "fitted_calibration_offset": calibration_offset,
            **row,
        })

    weights = {
        BASELINE: dict(zip(EXPERTS, point_weights)),
        CALIBRATED_CONTROL: dict(zip(EXPERTS, point_weights)),
        CANDIDATE: dict(zip(EXPERTS, pair_weights)),
    }
    return rows, weights


def _report(frame: pd.DataFrame) -> None:
    metrics = [
        "auc", "ap", "logloss", "brier", "positive_logloss",
        "negative_logloss", "f1", "precision", "recall", "mcc",
        "threshold", "pos_rate", "calibration_slope", "ece_quantile",
        "fitted_calibration_scale",
    ]
    print("\n=== strict holdout metrics: mean +/- std ===")
    print(frame.groupby("loss")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
    pivot = frame.pivot(index="seed", columns="loss")
    for reference in (BASELINE, CALIBRATED_CONTROL):
        print(f"\n=== {CANDIDATE} - {reference}: paired deltas ===")
        for metric in metrics:
            delta = pivot[(metric, CANDIDATE)] - pivot[(metric, reference)]
            print(
                f"{metric:22s} mean {delta.mean():+.5f}  "
                f"positive {int((delta > 0).sum())}/{len(delta)}  "
                f"values {np.round(delta.values, 5)}")


def run(n_reps: int, seed_offset: int, test_size: float,
        output: str) -> pd.DataFrame:
    X, y, text, org, e4, columns = _inputs()
    records = []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, _ = _one_repeat(
            X, y, text, org, e4, columns, seed, test_size)
        records.extend(rows)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _report(frame)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=206)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_meta_losses_control_seed206_5rep.csv")
    args = parser.parse_args()
    run(args.n_reps, args.seed_offset, args.test_size, args.output)
