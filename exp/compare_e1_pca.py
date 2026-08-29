"""H56: strict holdout test of block-PCA-compressed E1 finance columns.

`documents/h52_h60_and_h2_leverage_20260826.md` found E1's raw FINANCE_COLS
contain two tight near-duplicate blocks (size: total/current/fixed assets,
liabilities, equity, capital; profit: operating/ordinary/net income,
operating CF) and REJECTed H54's fix (deleting 5 of the 10 columns down to a
single representative each) as neutral. This tests the literature-standard
alternative -- PCA (Hotelling 1933) on those same two blocks, which retains a
weighted combination of every column's signal instead of discarding most of
it -- via `exp/finance_pca_features.py`. Never run before (ranked #4 in the
project's own research plan, `documents/signate_information_quality_research.md`
section 24, but skipped after H52/H54 landed neutral).

Only E1's finance-block columns change; every other Expert, the outer split,
inner folds, meta loss (plain logloss), alpha grid, cross-fitted Platt
calibration, and half_f1 threshold rule are shared with production. E1's own
LightGBM training (`_strict_lgbm_predict`, unmodified) is unaffected -- this
is a feature-input change, not a loss-function change.

Screening example:
  python exp/compare_e1_pca.py --n-reps 5 --seed-offset 281
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
from compare_meta_loss_families import _fit_one_loss  # noqa: E402
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from expert_groups import (  # noqa: E402
    FINANCE_COLS, FINANCE_ISNA_COLS, H1_COLS, SCALE_COLS,
)
from finance_pca_features import (  # noqa: E402
    FINANCE_PROFIT_BLOCK, FINANCE_SIZE_BLOCK, e1_pca_frames,
)
from holdout_check import _stack, _strict_lgbm_predict, fit_predict_experts  # noqa: E402
from meta_loss_lab import LOSS_SPECS  # noqa: E402


LOGLOSS_SPEC = next(spec for spec in LOSS_SPECS if spec.name == "logloss")

_BLOCK_COLS = set(FINANCE_SIZE_BLOCK) | set(FINANCE_PROFIT_BLOCK)
_E1_OTHER_COLS = (
    [c for c in FINANCE_COLS if c not in _BLOCK_COLS]
    + list(SCALE_COLS) + list(H1_COLS) + list(FINANCE_ISNA_COLS)
)


def _fit_predict_e1_pca(X, y, tr, target, seed) -> np.ndarray:
    Xtr, Xta = X.iloc[tr], X.iloc[target]
    Xtr_e1, Xta_e1 = e1_pca_frames(Xtr, Xta, _E1_OTHER_COLS, seed=seed)
    return _strict_lgbm_predict(Xtr_e1, y[tr], Xta_e1, seed)


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

    Z_train_baseline = _stack(expert_oof, EXPERTS)
    Z_holdout_baseline = _stack(expert_holdout, EXPERTS)
    rows = [{
        "seed": seed,
        "config": "e1_baseline_raw",
        **_fit_one_loss(
            Z_train_baseline, y_train, Z_holdout_baseline, y_holdout,
            LOGLOSS_SPEC, folds),
    }]

    pca_e1_oof = np.zeros(len(train_index))
    for inner_train, inner_valid in folds:
        pca_e1_oof[inner_valid] = _fit_predict_e1_pca(
            X, y, train_index[inner_train], train_index[inner_valid], seed)
    pca_e1_holdout = _fit_predict_e1_pca(
        X, y, train_index, holdout_index, seed)

    oof_candidate = dict(expert_oof)
    oof_candidate["E1_finance"] = pca_e1_oof
    holdout_candidate = dict(expert_holdout)
    holdout_candidate["E1_finance"] = pca_e1_holdout
    Z_train_candidate = _stack(oof_candidate, EXPERTS)
    Z_holdout_candidate = _stack(holdout_candidate, EXPERTS)
    rows.append({
        "seed": seed,
        "config": "e1_block_pca",
        **_fit_one_loss(
            Z_train_candidate, y_train, Z_holdout_candidate, y_holdout,
            LOGLOSS_SPEC, folds),
    })
    return rows


def _report(frame: pd.DataFrame) -> None:
    metrics = ["auc", "ap", "logloss", "brier", "f1", "precision", "recall",
              "mcc", "outer_oracle_f1"]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("config")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda v: f"{v:.5f}"))

    pivot = frame.pivot(index="seed", columns="config")
    baseline = "e1_baseline_raw"
    candidate = "e1_block_pca"
    print(f"\n=== {candidate} - {baseline}: paired deltas ===")
    for metric in ("f1", "ap", "auc", "logloss", "brier", "mcc",
                   "outer_oracle_f1"):
        delta = pivot[(metric, candidate)] - pivot[(metric, baseline)]
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
    parser.add_argument("--seed-offset", type=int, default=281)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_e1_pca_seed281_5rep.csv")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output)
