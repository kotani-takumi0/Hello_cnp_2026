"""H66: strict holdout test of counterfactual-twin preprocessing.

The candidate is a ninth expert built from the five fold-safe margins in
``counterfactual_twin_features.py``.  Existing eight-expert predictions are
shared exactly between baseline and candidate.  Only the candidate expert set
and the resulting meta fit differ.

Protocol per repeat:

1. Split all 742 labelled rows into A/B; B labels remain sealed.
2. Build OOF predictions for all experts inside A.
3. Select meta regularisation, weights, and the half-F1 threshold on A only.
4. Refit experts on A and evaluate untouched B.

The twin head adds one further inner cross-fit when constructing its own five
training features.  Thus no target row supplies either its label or itself as
a twin.

Examples:
  .venv/bin/python exp/compare_counterfactual_twin.py --n-reps 1 --seed-offset 499
  .venv/bin/python exp/compare_counterfactual_twin.py --n-reps 5 --seed-offset 500
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from counterfactual_twin_features import (  # noqa: E402
    TWIN_EXPERT_NAME, TWIN_HEAD_C, TWIN_NEIGHBORS, fit_predict_twin,
    load_twin_embedding_views,
)
from decision import verdict  # noqa: E402
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from metric_suite import score_predictions  # noqa: E402

EXPERTS_PLUS = EXPERTS + (TWIN_EXPERT_NAME,)
BASELINE = "meta_8_baseline"
CANDIDATE = "meta_9_with_twin"
TWIN_ONLY = "twin_standalone"


def _band_metrics(y: np.ndarray, probability: np.ndarray,
                  band: np.ndarray) -> dict[str, float | int]:
    y_band = y[band]
    p_band = probability[band]
    if len(y_band) == 0 or len(np.unique(y_band)) < 2:
        return {"band_n": int(len(y_band)), "band_auc": np.nan,
                "band_ap": np.nan}
    return {
        "band_n": int(len(y_band)),
        "band_auc": float(roc_auc_score(y_band, p_band)),
        "band_ap": float(average_precision_score(y_band, p_band)),
    }


def _fit_meta_prediction(expert_oof, expert_holdout, members, y_train,
                         seed: int):
    train_stack = _stack(expert_oof, members)
    holdout_stack = _stack(expert_holdout, members)
    alpha, inner_ap = select_alpha(train_stack, y_train, seed=seed)
    weight, intercept = fit_meta(train_stack, y_train, alpha=alpha)
    return (
        apply_meta(train_stack, weight, intercept),
        apply_meta(holdout_stack, weight, intercept),
        alpha,
        inner_ap,
        weight,
    )


def _one_repeat(X, y, text, org, e4, columns, embeddings, seed: int,
                test_size: float, n_neighbors: int,
                head_c: float) -> tuple[list[dict], dict]:
    all_indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_idx], y[holdout_idx]
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(train_idx, y_train))

    expert_oof = {name: np.zeros(len(train_idx)) for name in EXPERTS_PLUS}
    for inner_train, inner_valid in folds:
        reference = train_idx[inner_train]
        target = train_idx[inner_valid]
        predictions = fit_predict_experts(
            X, y, text, org, reference, target, seed, columns,
            e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
            strict_lgbm=True,
        )
        for name in EXPERTS:
            expert_oof[name][inner_valid] = predictions[name]
        expert_oof[TWIN_EXPERT_NAME][inner_valid] = fit_predict_twin(
            X, y, embeddings, reference, target, seed,
            n_neighbors=n_neighbors, head_c=head_c,
        )

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
        strict_lgbm=True,
    )
    expert_holdout[TWIN_EXPERT_NAME] = fit_predict_twin(
        X, y, embeddings, train_idx, holdout_idx, seed,
        n_neighbors=n_neighbors, head_c=head_c,
    )

    base = _fit_meta_prediction(
        expert_oof, expert_holdout, EXPERTS, y_train, seed)
    candidate = _fit_meta_prediction(
        expert_oof, expert_holdout, EXPERTS_PLUS, y_train, seed)
    base_train, base_holdout = base[:2]
    # The gate is descriptive only: model fitting never sees B labels.  It is
    # fixed by the production model's measured uncertain band.
    band = (base_holdout > 0.1) & (base_holdout < 0.9)

    rows = []
    for label, members, result in (
        (BASELINE, EXPERTS, base),
        (CANDIDATE, EXPERTS_PLUS, candidate),
    ):
        train_probability, holdout_probability, alpha, inner_ap, weight = result
        threshold = _thresholds(
            train_probability, y_train, holdout_probability, folds,
            ("half_f1",),
        )["half_f1"]
        metrics = score_predictions(y_holdout, holdout_probability, threshold)
        rows.append({
            "seed": seed,
            "config": label,
            "alpha": alpha,
            "inner_ap": inner_ap,
            "twin_weight": (float(weight[-1])
                            if TWIN_EXPERT_NAME in members else np.nan),
            **_band_metrics(y_holdout, holdout_probability, band),
            **metrics,
        })

    twin_train = expert_oof[TWIN_EXPERT_NAME]
    twin_holdout = expert_holdout[TWIN_EXPERT_NAME]
    twin_threshold = _thresholds(
        twin_train, y_train, twin_holdout, folds, ("half_f1",)
    )["half_f1"]
    rows.append({
        "seed": seed,
        "config": TWIN_ONLY,
        "alpha": np.nan,
        "inner_ap": np.nan,
        "twin_weight": np.nan,
        **_band_metrics(y_holdout, twin_holdout, band),
        **score_predictions(y_holdout, twin_holdout, twin_threshold),
    })

    correlations = {
        name: float(spearmanr(twin_train, expert_oof[name]).statistic)
        for name in EXPERTS
    }
    diagnostics = {
        "correlations": correlations,
        "max_abs_spearman": max(abs(value) for value in correlations.values()),
        "candidate_weights": dict(zip(EXPERTS_PLUS, candidate[-1])),
    }
    return rows, diagnostics


def _report(frame: pd.DataFrame, diagnostics: list[dict]) -> None:
    metrics = [
        "auc", "ap", "f1", "precision", "recall", "mcc",
        "band_auc", "band_ap", "band_n", "threshold", "pos_rate",
    ]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("config")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))

    pivot = frame.pivot(index="seed", columns="config")
    print(f"\n=== {CANDIDATE} - {BASELINE}: paired deltas ===")
    for metric in ("auc", "ap", "f1", "band_auc", "band_ap"):
        delta = pivot[(metric, CANDIDATE)] - pivot[(metric, BASELINE)]
        print(
            f"{metric:10s} delta={delta.mean():+.5f}  "
            f"wins={int((delta > 1e-12).sum())}/"
            f"ties={int((np.abs(delta) <= 1e-12).sum())}/"
            f"losses={int((delta < -1e-12).sum())}  "
            f"values={np.round(delta.values, 5)}"
        )

    base = frame[frame.config == BASELINE].sort_values("seed")
    candidate = frame[frame.config == CANDIDATE].sort_values("seed")
    result, deltas = verdict(base, candidate)
    print(f"\n=== H66 formal screen ===")
    for metric in ("auc", "ap", "f1"):
        value = deltas[metric]
        print(
            f"{metric:4s} mean={value['mean']:+.5f}  "
            f"positive={value['n_pos']}/{value['n']}  "
            f"range=[{value['min']:+.5f}, {value['max']:+.5f}]"
        )
    print(f"verdict: {result}")

    print("\n=== twin diversity / use ===")
    for row, diag in zip(
        frame[frame.config == CANDIDATE].sort_values("seed").itertuples(),
        diagnostics,
    ):
        print(
            f"seed={row.seed} twin_weight={row.twin_weight:.4f} "
            f"max_abs_spearman={diag['max_abs_spearman']:.4f}"
        )
    mean_weight = frame.loc[frame.config == CANDIDATE, "twin_weight"].mean()
    print(f"mean twin meta weight: {mean_weight:.4f}")


def run(n_reps: int, seed_offset: int, test_size: float, output: str,
        n_neighbors: int = TWIN_NEIGHBORS,
        head_c: float = TWIN_HEAD_C) -> pd.DataFrame:
    X, y, text, org, e4, columns = _inputs()
    train_embeddings, _ = load_twin_embedding_views()
    records, diagnostics = [], []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, diag = _one_repeat(
            X, y, text, org, e4, columns, train_embeddings, seed,
            test_size, n_neighbors, head_c,
        )
        records.extend(rows)
        diagnostics.append(diag)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _report(frame, diagnostics)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--neighbors", type=int, default=TWIN_NEIGHBORS)
    parser.add_argument("--head-c", type=float, default=TWIN_HEAD_C)
    parser.add_argument(
        "--output",
        default="exp/holdout_counterfactual_twin_seed500_5rep.csv",
    )
    arguments = parser.parse_args()
    run(
        arguments.n_reps, arguments.seed_offset, arguments.test_size,
        arguments.output, arguments.neighbors, arguments.head_c,
    )
