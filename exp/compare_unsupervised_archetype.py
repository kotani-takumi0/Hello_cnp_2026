"""H65: strict holdout screen for an unsupervised archetype expert (E10).

`exp/archetype_features.py` fits a target-free GMM over 7 financial-health
ratios + 11 survey items, then a plain LR head on the K soft membership
probabilities -- a decision geometry (density-based soft clustering) none of
the current 8 production experts produce. Standalone strength is expected to
be weak (financial ratios + survey alone, no text, no LightGBM); the point is
whether it adds independent signal to the existing 8-expert meta blend, per
`documents/deep-research-report.md`'s own framing:
  Value ~= Performance x Residual Diversity x Stability.

Only the expert set changes (8 -> 9). Outer split, inner folds, expert
fitting for E0-E0b (`fit_predict_experts`, strict LightGBM), meta loss
(pointwise logistic, alpha selected inside A only), and the `half_f1`
threshold rule are all shared with production (mirrors
`compare_meta_losses.py` / `compare_e1_pca.py`).

Example:
  python exp/compare_unsupervised_archetype.py --n-reps 5 --seed-offset 300
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from archetype_features import fit_predict_archetype  # noqa: E402
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from metric_suite import score_predictions  # noqa: E402

E10_NAME = "E10_archetype"
EXPERTS_PLUS = EXPERTS + (E10_NAME,)
BASELINE = "meta_8_baseline"
CANDIDATE = "meta_9_with_archetype"


def _one_repeat(X, y, text, org, e4, columns, seed: int,
                test_size: float) -> tuple[list[dict], dict]:
    all_indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_idx], y[holdout_idx]
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(train_idx, y_train))

    expert_oof = {name: np.zeros(len(train_idx)) for name in EXPERTS_PLUS}
    for inner_train, inner_valid in folds:
        predictions = fit_predict_experts(
            X, y, text, org, train_idx[inner_train], train_idx[inner_valid],
            seed, columns, e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
            strict_lgbm=True,
        )
        for name in EXPERTS:
            expert_oof[name][inner_valid] = predictions[name]
        expert_oof[E10_NAME][inner_valid] = fit_predict_archetype(
            X.iloc[train_idx[inner_train]], y[train_idx[inner_train]],
            X.iloc[train_idx[inner_valid]], seed)

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True,
    )
    expert_holdout[E10_NAME] = fit_predict_archetype(
        X.iloc[train_idx], y[train_idx], X.iloc[holdout_idx], seed)

    rows = []
    w = b = None
    for label, members in ((BASELINE, EXPERTS), (CANDIDATE, EXPERTS_PLUS)):
        Za = _stack(expert_oof, members)
        Zb = _stack(expert_holdout, members)
        alpha, _ = select_alpha(Za, y_train, seed=seed)
        w, b = fit_meta(Za, y_train, alpha=alpha)
        pa = apply_meta(Za, w, b)
        pb = apply_meta(Zb, w, b)
        threshold = _thresholds(
            pa, y_train, pb, folds, ("half_f1",))["half_f1"]
        row = score_predictions(y_holdout, pb, threshold)
        rows.append({"seed": seed, "config": label, "alpha": alpha, **row})
    weights_candidate = dict(zip(EXPERTS_PLUS, w))
    return rows, weights_candidate


def _report(frame: pd.DataFrame) -> None:
    metrics = ["auc", "ap", "f1", "precision", "recall", "mcc"]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("config")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda v: f"{v:.5f}"))

    pivot = frame.pivot(index="seed", columns="config")
    print(f"\n=== {CANDIDATE} - {BASELINE}: paired deltas ===")
    for metric in metrics:
        delta = pivot[(metric, CANDIDATE)] - pivot[(metric, BASELINE)]
        print(
            f"{metric:10s} delta={delta.mean():+.5f}  "
            f"wins={int((delta > 1e-12).sum())}/"
            f"ties={int((np.abs(delta) <= 1e-12).sum())}/"
            f"losses={int((delta < -1e-12).sum())}")


def run(n_reps: int, seed_offset: int, test_size: float,
        output: str) -> pd.DataFrame:
    X, y, text, org, e4, columns = _inputs()
    records = []
    last_weights = None
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, weights = _one_repeat(
            X, y, text, org, e4, columns, seed, test_size)
        records.extend(rows)
        last_weights = weights
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _report(frame)
    if last_weights is not None:
        print(f"\n=== last-rep {CANDIDATE} meta weights ===")
        for n, v in sorted(last_weights.items(), key=lambda kv: -kv[1]):
            print(f"  {n:16s} {v:.4f}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output",
        default="exp/holdout_unsupervised_archetype_seed300_5rep.csv")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output)
