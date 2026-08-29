"""Strict holdout test of F1-approximation losses applied to a single Expert's
own LightGBM training objective, instead of the meta-blend layer.

The meta-blend layer already rejected four independent F1-approximation
losses (documents/sigmoidf1_loss_audit_20260827.md and its predecessors).
This script asks the same question one layer down: does replacing E0_anchor's
ordinary binary logloss with a custom sigmoidF1 or weighted-logloss objective
change the final 8-expert meta pipeline's holdout F1?  Every other Expert,
the outer split, inner folds, meta loss (plain non-negative logloss), alpha
grid, cross-fitted Platt calibration, and half_f1 threshold rule are shared
with the current production configuration so the comparison isolates E0's
training loss only.

Screening example:
  python exp/compare_expert_f1_loss.py --n-reps 5 --seed-offset 256
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_loss_families import _fit_one_loss  # noqa: E402
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from holdout_check import _stack, fit_predict_experts  # noqa: E402
from meta_loss_lab import LOSS_SPECS  # noqa: E402
from organization_features import (  # noqa: E402
    ORG_SCORE_COL, fold_org_preds,
)
from ensemble_experts import DX_SCORE_COL  # noqa: E402
from text_features import nested_text_pred  # noqa: E402
from harness import BASE_PARAMS  # noqa: E402
from lgbm_f1_objectives import (  # noqa: E402
    base_score, sigmoid_f1_feval, sigmoid_f1_fobj,
    weighted_logloss_feval, weighted_logloss_fobj,
)


LOGLOSS_SPEC = next(spec for spec in LOSS_SPECS if spec.name == "logloss")

CANDIDATES = {
    "e0_sigmoid_f1_b1e0": (sigmoid_f1_fobj(1.0, 0.0), sigmoid_f1_feval(1.0, 0.0)),
    "e0_sigmoid_f1_b7e1": (sigmoid_f1_fobj(7.0, 1.0), sigmoid_f1_feval(7.0, 1.0)),
    "e0_weighted_logloss_pos15": (
        weighted_logloss_fobj(1.5), weighted_logloss_feval(1.5)),
}

_CUSTOM_PARAMS = {k: v for k, v in BASE_PARAMS.items()
                  if k not in ("objective", "metric")}


def _strict_lgbm_predict_custom(Xtr, ytr, Xtarget, seed, fobj, feval):
    """Mirrors holdout_check._strict_lgbm_predict but with a custom fobj/feval.

    A custom objective bypasses LightGBM's automatic boost-from-average, so an
    explicit init_score (logit of the fit fold's positive rate) carries that
    role instead, and must be added back manually after ``predict`` (which
    returns only the summed tree contribution for a custom/None objective).
    """
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, stop_idx = next(inner.split(Xtr, ytr))
    params = {**_CUSTOM_PARAMS, "seed": seed, "objective": fobj}

    b_fit = base_score(ytr[fit_idx])
    dtr = lgb.Dataset(Xtr.iloc[fit_idx], ytr[fit_idx],
                      init_score=np.full(len(fit_idx), b_fit, dtype=float))
    dva = lgb.Dataset(Xtr.iloc[stop_idx], ytr[stop_idx],
                      init_score=np.full(len(stop_idx), b_fit, dtype=float),
                      reference=dtr)
    scout = lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    rounds = max(1, int(scout.best_iteration))

    b_full = base_score(ytr)
    dtr_full = lgb.Dataset(
        Xtr, ytr, init_score=np.full(len(ytr), b_full, dtype=float))
    final = lgb.train(params, dtr_full, num_boost_round=rounds)
    raw = final.predict(Xtarget, num_iteration=rounds)
    return expit(b_full + raw)


def _fit_predict_e0_custom(X, y, txt, org, tr, target, seed, c0, fobj, feval):
    Xtr, Xta = X.iloc[tr], X.iloc[target]
    t_tr, t_ta = nested_text_pred(txt[tr], y[tr], txt[target])
    o_tr, o_ta, _ = fold_org_preds(org[tr], y[tr], org[target], org[:1])
    A_tr, A_ta = Xtr[c0].copy(), Xta[c0].copy()
    A_tr[DX_SCORE_COL], A_ta[DX_SCORE_COL] = t_tr, t_ta
    A_tr[ORG_SCORE_COL], A_ta[ORG_SCORE_COL] = o_tr, o_ta
    return _strict_lgbm_predict_custom(A_tr, y[tr], A_ta, seed, fobj, feval)


def _one_repeat(X, y, text, org, e4, columns, seed: int,
                test_size: float) -> list[dict]:
    all_indices = np.arange(len(y))
    train_index, holdout_index = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_index], y[holdout_index]
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(train_index, y_train))
    c0 = columns[0]

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
        **_fit_one_loss(
            Z_train_baseline, y_train, Z_holdout_baseline, y_holdout,
            LOGLOSS_SPEC, folds),
        "expert_loss": "e0_baseline_logloss",
    }]

    for candidate_name, (fobj, feval) in CANDIDATES.items():
        candidate_e0_oof = np.zeros(len(train_index))
        for inner_train, inner_valid in folds:
            candidate_e0_oof[inner_valid] = _fit_predict_e0_custom(
                X, y, text, org, train_index[inner_train],
                train_index[inner_valid], seed, c0, fobj, feval)
        candidate_e0_holdout = _fit_predict_e0_custom(
            X, y, text, org, train_index, holdout_index, seed, c0, fobj, feval)

        oof_candidate = dict(expert_oof)
        oof_candidate["E0_anchor"] = candidate_e0_oof
        holdout_candidate = dict(expert_holdout)
        holdout_candidate["E0_anchor"] = candidate_e0_holdout
        Z_train_candidate = _stack(oof_candidate, EXPERTS)
        Z_holdout_candidate = _stack(holdout_candidate, EXPERTS)

        row = _fit_one_loss(
            Z_train_candidate, y_train, Z_holdout_candidate, y_holdout,
            LOGLOSS_SPEC, folds)
        row["expert_loss"] = candidate_name
        rows.append({"seed": seed, **row})
    return rows


def _report(frame: pd.DataFrame) -> None:
    metrics = ["auc", "ap", "logloss", "brier", "f1", "precision", "recall",
              "mcc", "outer_oracle_f1"]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("expert_loss")[metrics].agg(["mean", "std"]).T
         .to_string(float_format=lambda v: f"{v:.5f}"))

    pivot = frame.pivot(index="seed", columns="expert_loss")
    baseline = "e0_baseline_logloss"
    for candidate in CANDIDATES:
        print(f"\n=== {candidate} - {baseline}: paired deltas ===")
        for metric in ("f1", "ap", "logloss", "brier", "mcc",
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
    parser.add_argument("--seed-offset", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument(
        "--output", default="exp/holdout_expert_f1_loss_seed256_5rep.csv")
    arguments = parser.parse_args()
    run(arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output)
