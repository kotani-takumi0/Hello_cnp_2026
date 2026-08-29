"""Create the positive-weight 1.5 meta-LogLoss challenger submission.

This deploys the same learning procedure used by
``compare_meta_loss_families.py``:

1. select L2 alpha by five-fold AP on the train OOF expert logits;
2. obtain cross-fitted weighted-LogLoss meta margins;
3. fit an unregularized Platt layer on those cross-fitted margins;
4. fit the weighted meta learner on all train OOF rows;
5. apply the calibrated ``F1*/2`` threshold to test predictions.

Only the meta objective changes.  The eight experts and their cached seed-42
predictions are identical to the current H2+H7-both4 submission.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_loss_families import (  # noqa: E402
    _best_f1, _cross_fitted_margins, _fit_platt, _select_alpha,
    _standardized_prediction,
)
from ensemble_experts import compute_expert_preds, expert_names  # noqa: E402
from holdout_check import _stack  # noqa: E402
from meta_loss_lab import LOSS_SPECS, fit_loss_meta  # noqa: E402
from metric_suite import score_predictions  # noqa: E402


CURRENT_EXPERTS = expert_names(cross=True, linear=True)
CURRENT_LINEAR_C = 0.03
LOSS_NAME = "weighted_logloss_pos15"
DEFAULT_OUTPUT = (
    "submission/submission_ensemble_experts_seed42_aauto_concatemb_"
    "crosslr_lin003_h2h7both4v2eq_weightedloglosspos15_"
    "cvplatt_halff1.csv"
)


def run(output: str) -> None:
    y, expert_oof, expert_test = compute_expert_preds(
        42, cache=True, concat_embed=True, cross=True, e7_model="lr",
        linear=True, linear_c=CURRENT_LINEAR_C, h2h7_both4=True,
    )
    Z_oof = _stack(expert_oof, CURRENT_EXPERTS)
    Z_test = _stack(expert_test, CURRENT_EXPERTS)
    spec = next(spec for spec in LOSS_SPECS if spec.name == LOSS_NAME)
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42).split(Z_oof, y))

    alpha, inner_ap = _select_alpha(Z_oof, y, spec, folds)
    cross_fitted_margin = _cross_fitted_margins(
        Z_oof, y, spec, alpha, folds)
    calibration_scale, calibration_intercept = _fit_platt(
        cross_fitted_margin, y)
    calibrated_oof = expit(
        calibration_scale * cross_fitted_margin + calibration_intercept)

    weights, intercept = fit_loss_meta(Z_oof, y, spec, alpha)
    _, standardized_test_margin = _standardized_prediction(
        Z_oof, Z_test, weights, intercept)
    calibrated_test = expit(
        calibration_scale * standardized_test_margin + calibration_intercept)

    best_oof_f1, best_oof_threshold = _best_f1(y, calibrated_oof)
    threshold = best_oof_f1 / 2.0
    label = (calibrated_test >= threshold).astype(int)
    metrics = score_predictions(y, calibrated_oof, threshold)

    test = pd.read_csv("data/test.csv")
    sample = pd.read_csv(
        "data/sample_submit.csv", header=None,
        names=["企業ID", "購入フラグ"])
    prediction = pd.DataFrame({"企業ID": test["企業ID"], "pred": label})
    submission = sample[["企業ID"]].merge(
        prediction, on="企業ID", how="left", validate="one_to_one")
    assert submission["pred"].notna().all(), "missing submission predictions"
    assert len(submission) == len(test) == 800
    assert set(label) <= {0, 1}
    if os.path.exists(output):
        raise FileExistsError(
            f"refusing to overwrite existing submission: {output}")
    submission.assign(pred=submission["pred"].astype(int)).to_csv(
        output, index=False, header=False, lineterminator="\n")

    normalized_weights = weights / weights.sum()
    print(f"loss={LOSS_NAME}, alpha={alpha:g}, inner AP={inner_ap:.6f}")
    print(
        f"cross-fitted Platt scale={calibration_scale:.6f}, "
        f"offset={calibration_intercept:.6f}")
    print(
        f"cross-fitted OOF AUC/AP/F1={metrics['auc']:.4f}/"
        f"{metrics['ap']:.4f}/{metrics['f1']:.4f}")
    print(
        f"OOF best F1={best_oof_f1:.4f} at {best_oof_threshold:.6f}; "
        f"deployed half_f1 threshold={threshold:.6f}")
    print("normalized meta weights:")
    for name, weight in zip(CURRENT_EXPERTS, normalized_weights):
        print(f"  {name:16s} {weight:.6f}")
    print(f"saved: {output} positives={int(label.sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run(arguments.output)
