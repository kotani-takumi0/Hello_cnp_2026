"""H90: evaluate ordinary synthetic-generation trace features.

Three protocols are intentionally kept separate:

``screen``
    Fast full-data OOF screen using the frozen current-expert caches.  It also
    reports a contiguous five-block OOF score for the trace head.  This is a
    diagnostic only because the cached stack is not a strict outer holdout.

``strict``
    Repeated stratified A/B holdout.  All eight existing experts, the H90
    trace head, meta weights, alpha and the half-F1 threshold are learned from
    A; B labels are opened only for the final paired report.

``forward``
    Two forward-chaining splits in enterprise-ID order.  The full pipeline is
    trained on an earlier prefix and evaluated on the immediately following
    block.  This is deliberately harsh, but a supposed generation artifact
    that cannot cross the train/test direction must not be submitted.

Examples:
  .venv/bin/python exp/compare_generator_trace.py --mode screen
  .venv/bin/python exp/compare_generator_trace.py --mode strict --n-reps 5
  .venv/bin/python exp/compare_generator_trace.py --mode forward
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from decision import verdict  # noqa: E402
from ensemble_experts import THS, blend_oof  # noqa: E402
from generator_trace_features import (  # noqa: E402
    TRACE_EXPERT_NAME, TRACE_GROUP_PREFIXES, TRACE_HEAD_C,
    build_generator_trace_features, fit_predict_trace,
    select_trace_feature_group,
)
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from metric_suite import score_predictions  # noqa: E402


BASELINE = "meta_8_baseline"
CANDIDATE = "meta_9_with_generator_trace"
TRACE_ONLY = "generator_trace_standalone"
EXPERTS_PLUS = EXPERTS + (TRACE_EXPERT_NAME,)
CURRENT_CACHE = (
    "exp/_experts_seed{seed}_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
)


def _band_metrics(y: np.ndarray, probability: np.ndarray,
                  band: np.ndarray) -> dict[str, float | int]:
    y_band = np.asarray(y)[band]
    p_band = np.asarray(probability)[band]
    if len(y_band) == 0 or len(np.unique(y_band)) < 2:
        return {"band_n": int(len(y_band)), "band_auc": np.nan,
                "band_ap": np.nan}
    return {
        "band_n": int(len(y_band)),
        "band_auc": float(roc_auc_score(y_band, p_band)),
        "band_ap": float(average_precision_score(y_band, p_band)),
    }


def _half_f1_score_row(y: np.ndarray, probability: np.ndarray) -> dict:
    best_index = int(np.argmax([
        f1_score(y, probability >= threshold) for threshold in THS
    ]))
    achieved_f1 = f1_score(y, probability >= THS[best_index])
    return score_predictions(y, probability, achieved_f1 / 2.0)


def _crossfit_trace(trace: pd.DataFrame, y: np.ndarray, folds,
                    target_trace: pd.DataFrame | None = None,
                    C: float = TRACE_HEAD_C) -> tuple[np.ndarray,
                                                      np.ndarray | None]:
    oof = np.zeros(len(y), dtype=float)
    target_parts = []
    for train_indices, valid_indices in folds:
        oof[valid_indices] = fit_predict_trace(
            trace, y, train_indices, valid_indices, C=C)
        if target_trace is not None:
            joined = pd.concat([trace, target_trace], ignore_index=True)
            shifted_target = np.arange(len(trace), len(joined))
            # ``fit_predict_trace`` indexes one matrix.  Concatenation is safe:
            # the transformer is already row-local and only the head is fit.
            target_parts.append(fit_predict_trace(
                joined, y, train_indices, shifted_target, C=C))
    target = (np.mean(target_parts, axis=0) if target_parts else None)
    return oof, target


def _load_current_cache(seed: int):
    path = CURRENT_CACHE.format(seed=seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"current expert cache is missing: {path}")
    data = np.load(path)
    y = data["y"].astype(int)
    oof = {name: data[f"oof_{name}"] for name in EXPERTS}
    test = {name: data[f"te_{name}"] for name in EXPERTS}
    return y, oof, test


def _paired_report(frame: pd.DataFrame, title: str,
                   formal: bool = False) -> None:
    metrics = ["auc", "ap", "f1", "precision", "recall", "mcc",
               "band_auc", "band_ap", "threshold", "pos_rate"]
    available = [metric for metric in metrics if metric in frame]
    print(f"\n=== {title}: mean +/- std ===")
    print(frame.groupby("config")[available].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
    pivot = frame.pivot(index="split", columns="config")
    if BASELINE in pivot.columns.get_level_values(1) and \
            CANDIDATE in pivot.columns.get_level_values(1):
        print(f"\n=== {CANDIDATE} - {BASELINE} ===")
        for metric in ("auc", "ap", "f1", "band_auc", "band_ap"):
            if metric not in frame:
                continue
            delta = pivot[(metric, CANDIDATE)] - pivot[(metric, BASELINE)]
            print(
                f"{metric:10s} delta={delta.mean():+.5f}  "
                f"wins={int((delta > 1e-12).sum())}/"
                f"ties={int((np.abs(delta) <= 1e-12).sum())}/"
                f"losses={int((delta < -1e-12).sum())}  "
                f"values={np.round(delta.values, 5)}"
            )
        if formal:
            base = frame[frame.config == BASELINE].sort_values("split")
            candidate = frame[frame.config == CANDIDATE].sort_values("split")
            result, deltas = verdict(base, candidate)
            print("\n=== H90 formal screen ===")
            for metric in ("auc", "ap", "f1"):
                value = deltas[metric]
                print(
                    f"{metric:4s} mean={value['mean']:+.5f} "
                    f"positive={value['n_pos']}/{value['n']} "
                    f"range=[{value['min']:+.5f}, {value['max']:+.5f}]"
                )
            print(f"verdict: {result}")


def run_screen(n_reps: int, seed_offset: int, output: str,
               C: float = TRACE_HEAD_C,
               feature_group: str = "all") -> pd.DataFrame:
    raw_train = pd.read_csv("data/train.csv")
    raw_test = pd.read_csv("data/test.csv")
    trace = select_trace_feature_group(
        build_generator_trace_features(raw_train), feature_group)
    trace_test = select_trace_feature_group(
        build_generator_trace_features(raw_test), feature_group)
    y_raw = raw_train["購入フラグ"].to_numpy(dtype=int)
    records = []

    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"screen {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        y, expert_oof, expert_test = _load_current_cache(seed)
        if not np.array_equal(y, y_raw):
            raise RuntimeError("cache labels do not match data/train.csv")
        folds = list(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=seed).split(trace, y))
        trace_oof, trace_te = _crossfit_trace(
            trace, y, folds, trace_test, C=C)

        base, _, _, _ = blend_oof(
            y, expert_oof, expert_test, seed, experts=EXPERTS)
        candidate_oof = {**expert_oof, TRACE_EXPERT_NAME: trace_oof}
        candidate_test = {**expert_test, TRACE_EXPERT_NAME: trace_te}
        candidate, _, weights, _ = blend_oof(
            y, candidate_oof, candidate_test, seed, experts=EXPERTS_PLUS)
        band = (base > 0.1) & (base < 0.9)

        correlations = [
            abs(float(spearmanr(trace_oof, expert_oof[name]).statistic))
            for name in EXPERTS
        ]
        for label, probability in (
            (BASELINE, base),
            (CANDIDATE, candidate),
            (TRACE_ONLY, trace_oof),
        ):
            records.append({
                "mode": "screen_random_oof",
                "feature_group": feature_group,
                "split": f"seed_{seed}",
                "config": label,
                "trace_weight": (float(weights[-1])
                                 if label == CANDIDATE else np.nan),
                "trace_max_abs_spearman": max(correlations),
                **_band_metrics(y, probability, band),
                **_half_f1_score_row(y, probability),
            })

    # One label-free contiguous-block construction.  It is deliberately not
    # repeated because neither the folds nor the LR head contain randomness.
    blocked_folds = list(KFold(n_splits=5, shuffle=False).split(trace))
    blocked_oof, _ = _crossfit_trace(trace, y_raw, blocked_folds, C=C)
    records.append({
        "mode": "screen_contiguous_blocks",
        "feature_group": feature_group,
        "split": "blocked_5fold",
        "config": TRACE_ONLY,
        "trace_weight": np.nan,
        "trace_max_abs_spearman": np.nan,
        "band_n": np.nan,
        "band_auc": np.nan,
        "band_ap": np.nan,
        **_half_f1_score_row(y_raw, blocked_oof),
    })

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    random_frame = frame[frame["mode"] == "screen_random_oof"]
    _paired_report(random_frame, "H90 cached full-data screen")
    blocked = frame[frame["mode"] == "screen_contiguous_blocks"].iloc[0]
    print("\n=== trace-only contiguous five-block OOF ===")
    print(
        f"AUC={blocked.auc:.5f} AP={blocked.ap:.5f} "
        f"F1(half_f1)={blocked.f1:.5f}"
    )
    print(
        "trace feature count="
        f"{trace.shape[1]}, group={feature_group}, C={C:g}, "
        f"constant_on_train={(trace.nunique() == 1).sum()}"
    )
    return frame


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


def _evaluate_outer_split(X, y, text, org, e4, columns,
                          trace: pd.DataFrame, raw_ids: np.ndarray,
                          train_idx: np.ndarray, holdout_idx: np.ndarray,
                          seed: int, split: str,
                          C: float = TRACE_HEAD_C) -> tuple[list[dict], dict]:
    train_idx = np.asarray(train_idx, dtype=int)
    holdout_idx = np.asarray(holdout_idx, dtype=int)
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
        expert_oof[TRACE_EXPERT_NAME][inner_valid] = fit_predict_trace(
            trace, y, reference, target, C=C)

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
        strict_lgbm=True,
    )
    expert_holdout[TRACE_EXPERT_NAME] = fit_predict_trace(
        trace, y, train_idx, holdout_idx, C=C)

    baseline = _fit_meta_prediction(
        expert_oof, expert_holdout, EXPERTS, y_train, seed)
    candidate = _fit_meta_prediction(
        expert_oof, expert_holdout, EXPERTS_PLUS, y_train, seed)
    baseline_holdout = baseline[1]
    band = (baseline_holdout > 0.1) & (baseline_holdout < 0.9)

    rows = []
    for label, members, fitted in (
        (BASELINE, EXPERTS, baseline),
        (CANDIDATE, EXPERTS_PLUS, candidate),
    ):
        train_probability, holdout_probability, alpha, inner_ap, weight = fitted
        threshold = _thresholds(
            train_probability, y_train, holdout_probability, folds,
            ("half_f1",),
        )["half_f1"]
        rows.append({
            "split": split,
            "seed": seed,
            "config": label,
            "train_n": len(train_idx),
            "holdout_n": len(holdout_idx),
            "train_id_min": int(raw_ids[train_idx].min()),
            "train_id_max": int(raw_ids[train_idx].max()),
            "holdout_id_min": int(raw_ids[holdout_idx].min()),
            "holdout_id_max": int(raw_ids[holdout_idx].max()),
            "alpha": alpha,
            "inner_ap": inner_ap,
            "trace_weight": (float(weight[-1])
                             if TRACE_EXPERT_NAME in members else np.nan),
            **_band_metrics(y_holdout, holdout_probability, band),
            **score_predictions(y_holdout, holdout_probability, threshold),
        })

    trace_train = expert_oof[TRACE_EXPERT_NAME]
    trace_holdout = expert_holdout[TRACE_EXPERT_NAME]
    trace_threshold = _thresholds(
        trace_train, y_train, trace_holdout, folds, ("half_f1",)
    )["half_f1"]
    rows.append({
        "split": split,
        "seed": seed,
        "config": TRACE_ONLY,
        "train_n": len(train_idx),
        "holdout_n": len(holdout_idx),
        "train_id_min": int(raw_ids[train_idx].min()),
        "train_id_max": int(raw_ids[train_idx].max()),
        "holdout_id_min": int(raw_ids[holdout_idx].min()),
        "holdout_id_max": int(raw_ids[holdout_idx].max()),
        "alpha": np.nan,
        "inner_ap": np.nan,
        "trace_weight": np.nan,
        **_band_metrics(y_holdout, trace_holdout, band),
        **score_predictions(y_holdout, trace_holdout, trace_threshold),
    })

    correlations = {
        name: float(spearmanr(trace_train, expert_oof[name]).statistic)
        for name in EXPERTS
    }
    diagnostics = {
        "split": split,
        "trace_weight": float(candidate[-1][-1]),
        "max_abs_spearman": max(abs(value) for value in correlations.values()),
        "correlations": correlations,
    }
    return rows, diagnostics


def _load_strict_inputs():
    X, y, text, org, e4, columns = _inputs()
    raw = pd.read_csv("data/train.csv")
    trace = build_generator_trace_features(raw)
    if len(raw) != len(X) or not np.array_equal(
            y, raw["購入フラグ"].to_numpy(dtype=int)):
        raise RuntimeError("strict inputs are not aligned")
    return X, y, text, org, e4, columns, trace, raw["企業ID"].to_numpy()


def run_strict(n_reps: int, seed_offset: int, test_size: float,
               output: str, C: float = TRACE_HEAD_C) -> pd.DataFrame:
    X, y, text, org, e4, columns, trace, raw_ids = _load_strict_inputs()
    records, diagnostics = [], []
    all_indices = np.arange(len(y))
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"strict {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        train_idx, holdout_idx = train_test_split(
            all_indices, test_size=test_size, stratify=y,
            random_state=seed)
        rows, diag = _evaluate_outer_split(
            X, y, text, org, e4, columns, trace, raw_ids,
            train_idx, holdout_idx, seed, f"random_{seed}", C=C)
        records.extend(rows)
        diagnostics.append(diag)

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _paired_report(frame, "H90 strict outer holdout", formal=True)
    print("\n=== trace diversity / use ===")
    for diagnostic in diagnostics:
        print(
            f"split={diagnostic['split']} "
            f"weight={diagnostic['trace_weight']:.5f} "
            f"max_abs_spearman={diagnostic['max_abs_spearman']:.5f}"
        )
    return frame


def run_forward(output: str, seed_offset: int,
                C: float = TRACE_HEAD_C) -> pd.DataFrame:
    X, y, text, org, e4, columns, trace, raw_ids = _load_strict_inputs()
    order = np.argsort(raw_ids, kind="stable")
    n_rows = len(order)
    cuts = (n_rows // 2, 3 * n_rows // 4, n_rows)
    specifications = (
        (order[:cuts[0]], order[cuts[0]:cuts[1]], "forward_50_to_75"),
        (order[:cuts[1]], order[cuts[1]:cuts[2]], "forward_75_to_100"),
    )
    records, diagnostics = [], []
    for index, (train_idx, holdout_idx, label) in enumerate(specifications):
        seed = seed_offset + index
        print(
            f"{label}: train={len(train_idx)} holdout={len(holdout_idx)} "
            f"seed={seed}", flush=True)
        rows, diag = _evaluate_outer_split(
            X, y, text, org, e4, columns, trace, raw_ids,
            train_idx, holdout_idx, seed, label, C=C)
        records.extend(rows)
        diagnostics.append(diag)

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _paired_report(frame, "H90 ID-ordered forward validation")
    print("\n=== forward trace diversity / use ===")
    for diagnostic in diagnostics:
        print(
            f"split={diagnostic['split']} "
            f"weight={diagnostic['trace_weight']:.5f} "
            f"max_abs_spearman={diagnostic['max_abs_spearman']:.5f}"
        )
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("screen", "strict", "forward"),
                        default="screen")
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--head-c", type=float, default=TRACE_HEAD_C)
    parser.add_argument(
        "--feature-group",
        choices=("all", *TRACE_GROUP_PREFIXES),
        default="all",
        help="screen mode only: target-free predeclared feature family",
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()

    if arguments.mode == "screen":
        seed_offset = 0 if arguments.seed_offset is None else arguments.seed_offset
        group_tag = "" if arguments.feature_group == "all" else f"_{arguments.feature_group}"
        output = arguments.output or f"exp/generator_trace_screen{group_tag}_seed0_5rep.csv"
        run_screen(
            arguments.n_reps, seed_offset, output, C=arguments.head_c,
            feature_group=arguments.feature_group,
        )
    elif arguments.mode == "strict":
        seed_offset = 700 if arguments.seed_offset is None else arguments.seed_offset
        output = arguments.output or "exp/holdout_generator_trace_seed700_5rep.csv"
        run_strict(arguments.n_reps, seed_offset, arguments.test_size, output,
                   C=arguments.head_c)
    else:
        seed_offset = 900 if arguments.seed_offset is None else arguments.seed_offset
        output = arguments.output or "exp/forward_generator_trace_seed900.csv"
        run_forward(output, seed_offset, C=arguments.head_c)
