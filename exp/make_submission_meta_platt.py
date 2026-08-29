"""Create the H64 MLE-Platt calibrated challenger submission.

The model, expert set, meta weights, and ranking are identical to the current
H2+H7-both4 seed-42 submission.  Only a two-parameter, unregularized Platt
temperature is fitted on meta OOF logits before applying the existing
``half_f1`` decision rule.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import (  # noqa: E402
    _scores, blend_oof, compute_expert_preds, expert_names,
)
from meta_blend import to_logit  # noqa: E402
from meta_pairwise import fit_platt  # noqa: E402

CURRENT_EXPERTS = expert_names(cross=True, linear=True)
CURRENT_LINEAR_C = 0.03
DEFAULT_OUTPUT = (
    "submission/submission_ensemble_experts_seed42_aauto_concatemb_"
    "crosslr_lin003_h2h7both4v2eq_plattmle_halff1.csv"
)


def run(output: str) -> None:
    y, oof, test_predictions = compute_expert_preds(
        42, cache=True, concat_embed=True, cross=True, e7_model="lr",
        linear=True, linear_c=CURRENT_LINEAR_C, h2h7_both4=True,
    )
    raw_oof, raw_test, _, _ = blend_oof(
        y, oof, test_predictions, 42, alpha=None, experts=CURRENT_EXPERTS)

    scale, offset = fit_platt(to_logit(raw_oof), y)
    calibrated_oof = expit(scale * to_logit(raw_oof) + offset)
    calibrated_test = expit(scale * to_logit(raw_test) + offset)
    raw_score = _scores(y, raw_oof)
    calibrated_score = _scores(y, calibrated_oof)
    threshold = calibrated_score["f1"] / 2.0
    label = (calibrated_test >= threshold).astype(int)

    test = pd.read_csv("data/test.csv")
    sample = pd.read_csv(
        "data/sample_submit.csv", header=None, names=["企業ID", "購入フラグ"])
    prediction = pd.DataFrame({"企業ID": test["企業ID"], "pred": label})
    submission = sample[["企業ID"]].merge(prediction, on="企業ID", how="left")
    assert submission["pred"].notna().all(), "missing submission predictions"
    if os.path.exists(output):
        raise FileExistsError(f"refusing to overwrite existing submission: {output}")
    submission.assign(pred=submission["pred"].astype(int)).to_csv(
        output, index=False, header=False, lineterminator="\n")

    raw_equivalent_threshold = float(expit(
        (np.log(threshold / (1.0 - threshold)) - offset) / scale))
    print(f"Platt scale={scale:.6f}, offset={offset:.6f}")
    print(
        f"OOF AUC/AP/F1={calibrated_score['auc']:.4f}/"
        f"{calibrated_score['ap']:.4f}/{calibrated_score['f1']:.4f}")
    print(
        f"half_f1 calibrated threshold={threshold:.6f} "
        f"(raw equivalent={raw_equivalent_threshold:.6f}, "
        f"current raw half_f1={raw_score['f1'] / 2.0:.6f})")
    print(f"saved: {output} positives={int(label.sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)
