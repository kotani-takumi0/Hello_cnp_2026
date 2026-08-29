"""H67: compare anti-template DX text preprocessing with the current E3.

Two deliberately separate protocols are exposed through ``--mode``:

``screen``
    Repeated five-fold OOF comparison of the unchanged full-text E3 and the
    anti-template text expert.  This isolates whether sentence selection
    improves the text signal itself.

``strict``
    Repeated A/B holdout comparison of (a) the current eight experts,
    (b) replacing E3 with anti-template text, and (c) adding anti-template text
    as a ninth expert.  Existing expert predictions are shared exactly among
    all configurations; B labels are used only for the final report.

Examples:
  .venv/bin/python exp/compare_anti_template_text.py --mode screen
  .venv/bin/python exp/compare_anti_template_text.py --mode strict --n-reps 1 --seed-offset 599
  .venv/bin/python exp/compare_anti_template_text.py --mode strict --n-reps 5 --seed-offset 600
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
from anti_template_text_features import (  # noqa: E402
    ANTI_TEMPLATE_NAME, AntiTemplateSelector, fit_predict_anti_template,
)
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from decision import verdict  # noqa: E402
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from metric_suite import score_predictions  # noqa: E402
from text_features import build_model  # noqa: E402


BASELINE = "meta_8_baseline"
REPLACE = "meta_8_replace_e3"
ADD = "meta_9_add_anti"
ANTI_ONLY = "anti_template_standalone"
FULL_TEXT = "E3_full_text"
REPLACE_MEMBERS = tuple(
    ANTI_TEMPLATE_NAME if name == "E3_dx_text" else name for name in EXPERTS
)
ADD_MEMBERS = EXPERTS + (ANTI_TEMPLATE_NAME,)


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


def run_screen(n_reps: int, seed_offset: int, output: str) -> pd.DataFrame:
    X, y, text, _, _, _ = _inputs()
    records = []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"screen {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        folds = list(StratifiedKFold(
            n_splits=5, shuffle=True, random_state=seed).split(X, y))
        full_oof = np.zeros(len(y))
        anti_oof = np.zeros(len(y))
        for train_indices, valid_indices in folds:
            full = build_model().fit(text[train_indices], y[train_indices])
            full_oof[valid_indices] = full.predict_proba(
                text[valid_indices])[:, 1]
            anti_oof[valid_indices] = fit_predict_anti_template(
                X, y, text, train_indices, valid_indices)[0]

        correlation = float(spearmanr(full_oof, anti_oof).statistic)
        for model, probability in (
            (FULL_TEXT, full_oof),
            (ANTI_TEMPLATE_NAME, anti_oof),
        ):
            records.append({
                "seed": seed,
                "model": model,
                "auc": float(roc_auc_score(y, probability)),
                "ap": float(average_precision_score(y, probability)),
                "spearman_with_full": (1.0 if model == FULL_TEXT
                                       else correlation),
            })

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    print("\n=== text-only repeated OOF mean +/- std ===")
    print(frame.groupby("model")[["auc", "ap", "spearman_with_full"]]
          .agg(["mean", "std"]).T.to_string(
              float_format=lambda value: f"{value:.5f}"))
    pivot = frame.pivot(index="seed", columns="model")
    print(f"\n=== {ANTI_TEMPLATE_NAME} - {FULL_TEXT} ===")
    for metric in ("auc", "ap"):
        delta = (pivot[(metric, ANTI_TEMPLATE_NAME)]
                 - pivot[(metric, FULL_TEXT)])
        print(
            f"{metric:4s} delta={delta.mean():+.5f} "
            f"wins={int((delta > 0).sum())}/{len(delta)} "
            f"values={np.round(delta.values, 5)}"
        )
    return frame


def _fit_meta_prediction(expert_oof, expert_holdout, members, y_train,
                         seed: int):
    train_stack = _stack(expert_oof, members)
    holdout_stack = _stack(expert_holdout, members)
    alpha, inner_ap = select_alpha(train_stack, y_train, seed=seed)
    weights, intercept = fit_meta(train_stack, y_train, alpha=alpha)
    return (
        apply_meta(train_stack, weights, intercept),
        apply_meta(holdout_stack, weights, intercept),
        alpha,
        inner_ap,
        weights,
    )


def _one_strict_repeat(X, y, text, org, e4, columns, seed: int,
                       test_size: float) -> tuple[list[dict], dict]:
    all_indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_idx], y[holdout_idx]
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    ).split(train_idx, y_train))

    all_members = ADD_MEMBERS
    expert_oof = {name: np.zeros(len(train_idx)) for name in all_members}
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
        expert_oof[ANTI_TEMPLATE_NAME][inner_valid] = (
            fit_predict_anti_template(X, y, text, reference, target)[0]
        )

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
        strict_lgbm=True,
    )
    anti_result = fit_predict_anti_template(
        X, y, text, train_idx, holdout_idx)
    expert_holdout[ANTI_TEMPLATE_NAME] = anti_result[0]

    configurations = (
        (BASELINE, EXPERTS),
        (REPLACE, REPLACE_MEMBERS),
        (ADD, ADD_MEMBERS),
    )
    fitted = {
        label: _fit_meta_prediction(
            expert_oof, expert_holdout, members, y_train, seed)
        for label, members in configurations
    }
    baseline_holdout = fitted[BASELINE][1]
    band = (baseline_holdout > 0.1) & (baseline_holdout < 0.9)

    rows = []
    for label, members in configurations:
        train_probability, holdout_probability, alpha, inner_ap, weights = (
            fitted[label]
        )
        threshold = _thresholds(
            train_probability, y_train, holdout_probability, folds,
            ("half_f1",),
        )["half_f1"]
        weight_map = dict(zip(members, weights))
        rows.append({
            "seed": seed,
            "config": label,
            "alpha": alpha,
            "inner_ap": inner_ap,
            "e3_weight": float(weight_map.get("E3_dx_text", np.nan)),
            "anti_weight": float(weight_map.get(ANTI_TEMPLATE_NAME, np.nan)),
            **_band_metrics(y_holdout, holdout_probability, band),
            **score_predictions(y_holdout, holdout_probability, threshold),
        })

    anti_train = expert_oof[ANTI_TEMPLATE_NAME]
    anti_holdout = expert_holdout[ANTI_TEMPLATE_NAME]
    anti_threshold = _thresholds(
        anti_train, y_train, anti_holdout, folds, ("half_f1",)
    )["half_f1"]
    rows.append({
        "seed": seed,
        "config": ANTI_ONLY,
        "alpha": np.nan,
        "inner_ap": np.nan,
        "e3_weight": np.nan,
        "anti_weight": np.nan,
        **_band_metrics(y_holdout, anti_holdout, band),
        **score_predictions(y_holdout, anti_holdout, anti_threshold),
    })

    selector, selected_train, selected_holdout = anti_result[1:]
    train_diag = AntiTemplateSelector.diagnostics(
        text[train_idx], selected_train)
    holdout_diag = AntiTemplateSelector.diagnostics(
        text[holdout_idx], selected_holdout)
    correlations = {
        name: float(spearmanr(anti_train, expert_oof[name]).statistic)
        for name in EXPERTS
    }
    diagnostics = {
        "seed": seed,
        "max_abs_spearman": max(abs(value) for value in correlations.values()),
        "correlations": correlations,
        "train_character_retention": train_diag.character_retention,
        "holdout_character_retention": holdout_diag.character_retention,
    }
    return rows, diagnostics


def _formal_report(frame: pd.DataFrame, candidate_label: str) -> None:
    base = frame[frame.config == BASELINE].sort_values("seed")
    candidate = frame[frame.config == candidate_label].sort_values("seed")
    result, deltas = verdict(base, candidate)
    print(f"\n=== H67 formal screen: {candidate_label} ===")
    for metric in ("auc", "ap", "f1"):
        value = deltas[metric]
        print(
            f"{metric:4s} mean={value['mean']:+.5f} "
            f"positive={value['n_pos']}/{value['n']} "
            f"range=[{value['min']:+.5f}, {value['max']:+.5f}]"
        )
    print(f"verdict: {result}")


def _strict_report(frame: pd.DataFrame, diagnostics: list[dict]) -> None:
    metrics = [
        "auc", "ap", "f1", "precision", "recall", "mcc",
        "band_auc", "band_ap", "band_n", "threshold", "pos_rate",
        "e3_weight", "anti_weight",
    ]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("config")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
    pivot = frame.pivot(index="seed", columns="config")
    for candidate in (REPLACE, ADD):
        print(f"\n=== {candidate} - {BASELINE}: paired deltas ===")
        for metric in ("auc", "ap", "f1", "band_auc", "band_ap"):
            delta = pivot[(metric, candidate)] - pivot[(metric, BASELINE)]
            print(
                f"{metric:10s} delta={delta.mean():+.5f} "
                f"wins={int((delta > 1e-12).sum())}/"
                f"ties={int((np.abs(delta) <= 1e-12).sum())}/"
                f"losses={int((delta < -1e-12).sum())} "
                f"values={np.round(delta.values, 5)}"
            )
        _formal_report(frame, candidate)

    print("\n=== anti-template diversity / compression ===")
    for diag in diagnostics:
        print(
            f"seed={diag['seed']} "
            f"max_abs_spearman={diag['max_abs_spearman']:.4f} "
            f"retention(train/B)={diag['train_character_retention']:.3f}/"
            f"{diag['holdout_character_retention']:.3f}"
        )


def run_strict(n_reps: int, seed_offset: int, test_size: float,
               output: str) -> pd.DataFrame:
    X, y, text, org, e4, columns = _inputs()
    records, diagnostics = [], []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"strict {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, diag = _one_strict_repeat(
            X, y, text, org, e4, columns, seed, test_size)
        records.extend(rows)
        diagnostics.append(diag)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    _strict_report(frame, diagnostics)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("screen", "strict"),
                        default="screen")
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--output")
    arguments = parser.parse_args()

    if arguments.mode == "screen":
        seed_offset = (670 if arguments.seed_offset is None
                       else arguments.seed_offset)
        output = (arguments.output
                  or "exp/oof_anti_template_text_seed670_5rep.csv")
        run_screen(arguments.n_reps, seed_offset, output)
    else:
        seed_offset = (600 if arguments.seed_offset is None
                       else arguments.seed_offset)
        output = (arguments.output
                  or "exp/holdout_anti_template_text_seed600_5rep.csv")
        run_strict(
            arguments.n_reps, seed_offset, arguments.test_size, output)
