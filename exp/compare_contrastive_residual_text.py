"""H69: strict evaluation of the H67-minus-E3 signed residual expert.

The text-only screen measures full E3, hard anti-template H67, and the nested
contrastive residual.  The strict test shares all predictions and compares the
current eight experts against adding raw H67 or adding H69.

Examples:
  .venv/bin/python exp/compare_contrastive_residual_text.py --mode screen
  .venv/bin/python exp/compare_contrastive_residual_text.py --mode strict --n-reps 1 --seed-offset 619
  .venv/bin/python exp/compare_contrastive_residual_text.py --mode strict --n-reps 5 --seed-offset 620
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
    ANTI_TEMPLATE_NAME, fit_predict_anti_template,
)
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from contrastive_residual_text_features import (  # noqa: E402
    CONTRASTIVE_RESIDUAL_NAME, fit_apply_residual,
    fit_predict_contrastive_residual,
)
from decision import verdict  # noqa: E402
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha  # noqa: E402
from metric_suite import score_predictions  # noqa: E402
from text_features import build_model  # noqa: E402


BASELINE = "meta_8_baseline"
HARD_ADD = "meta_9_add_hard_h67"
RESIDUAL_ADD = "meta_9_add_residual_h69"
RESIDUAL_ONLY = "contrastive_residual_standalone"
FULL_TEXT = "E3_full_text"
HARD_MEMBERS = EXPERTS + (ANTI_TEMPLATE_NAME,)
RESIDUAL_MEMBERS = EXPERTS + (CONTRASTIVE_RESIDUAL_NAME,)
ALL_MEMBERS = EXPERTS + (ANTI_TEMPLATE_NAME, CONTRASTIVE_RESIDUAL_NAME)


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
        predictions = {
            FULL_TEXT: np.zeros(len(y)),
            ANTI_TEMPLATE_NAME: np.zeros(len(y)),
            CONTRASTIVE_RESIDUAL_NAME: np.zeros(len(y)),
        }
        diagnostics = []
        for train_indices, valid_indices in folds:
            full_model = build_model().fit(
                text[train_indices], y[train_indices])
            full_target = full_model.predict_proba(text[valid_indices])[:, 1]
            hard_target = fit_predict_anti_template(
                X, y, text, train_indices, valid_indices)[0]
            residual_result = fit_predict_contrastive_residual(
                X, y, text, train_indices, valid_indices,
                target_full_probability=full_target,
                target_hard_probability=hard_target,
            )
            predictions[FULL_TEXT][valid_indices] = full_target
            predictions[ANTI_TEMPLATE_NAME][valid_indices] = hard_target
            predictions[CONTRASTIVE_RESIDUAL_NAME][valid_indices] = (
                residual_result[0])
            diagnostics.append(residual_result[1])

        residual = predictions[CONTRASTIVE_RESIDUAL_NAME]
        full_correlation = float(spearmanr(
            residual, predictions[FULL_TEXT]).statistic)
        hard_correlation = float(spearmanr(
            residual, predictions[ANTI_TEMPLATE_NAME]).statistic)
        mean_calibrator = float(np.mean([
            diag.calibrator_coefficient for diag in diagnostics]))
        mean_abs_calibrator = float(np.mean([
            abs(diag.calibrator_coefficient) for diag in diagnostics]))
        for model, probability in predictions.items():
            records.append({
                "seed": seed,
                "model": model,
                "auc": float(roc_auc_score(y, probability)),
                "ap": float(average_precision_score(y, probability)),
                "residual_full_spearman": full_correlation,
                "residual_hard_spearman": hard_correlation,
                "mean_calibrator_coefficient": mean_calibrator,
                "mean_abs_calibrator_coefficient": mean_abs_calibrator,
            })

    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    metrics = [
        "auc", "ap", "residual_full_spearman",
        "residual_hard_spearman", "mean_calibrator_coefficient",
        "mean_abs_calibrator_coefficient",
    ]
    print("\n=== text-only repeated OOF mean +/- std ===")
    print(frame.groupby("model")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
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

    expert_oof = {name: np.zeros(len(train_idx)) for name in ALL_MEMBERS}
    fold_diagnostics = []
    for inner_train, inner_valid in folds:
        reference = train_idx[inner_train]
        target = train_idx[inner_valid]
        current = fit_predict_experts(
            X, y, text, org, reference, target, seed, columns,
            e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
            strict_lgbm=True,
        )
        for name in EXPERTS:
            expert_oof[name][inner_valid] = current[name]
        hard_target = fit_predict_anti_template(
            X, y, text, reference, target)[0]
        expert_oof[ANTI_TEMPLATE_NAME][inner_valid] = hard_target
        residual_result = fit_predict_contrastive_residual(
            X, y, text, reference, target,
            target_full_probability=current["E3_dx_text"],
            target_hard_probability=hard_target,
        )
        expert_oof[CONTRASTIVE_RESIDUAL_NAME][inner_valid] = (
            residual_result[0])
        fold_diagnostics.append(residual_result[1])

    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
        strict_lgbm=True,
    )
    expert_holdout[ANTI_TEMPLATE_NAME] = fit_predict_anti_template(
        X, y, text, train_idx, holdout_idx)[0]
    residual_holdout = fit_apply_residual(
        expert_oof["E3_dx_text"], expert_oof[ANTI_TEMPLATE_NAME], y_train,
        expert_holdout["E3_dx_text"], expert_holdout[ANTI_TEMPLATE_NAME],
    )
    expert_holdout[CONTRASTIVE_RESIDUAL_NAME] = residual_holdout[0]

    configurations = (
        (BASELINE, EXPERTS),
        (HARD_ADD, HARD_MEMBERS),
        (RESIDUAL_ADD, RESIDUAL_MEMBERS),
    )
    fitted = {
        label: _fit_meta_prediction(
            expert_oof, expert_holdout, members, y_train, seed)
        for label, members in configurations
    }
    band = ((fitted[BASELINE][1] > 0.1)
            & (fitted[BASELINE][1] < 0.9))

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
            "hard_weight": float(
                weight_map.get(ANTI_TEMPLATE_NAME, np.nan)),
            "residual_weight": float(
                weight_map.get(CONTRASTIVE_RESIDUAL_NAME, np.nan)),
            **_band_metrics(y_holdout, holdout_probability, band),
            **score_predictions(y_holdout, holdout_probability, threshold),
        })

    residual_train = expert_oof[CONTRASTIVE_RESIDUAL_NAME]
    residual_target = expert_holdout[CONTRASTIVE_RESIDUAL_NAME]
    residual_threshold = _thresholds(
        residual_train, y_train, residual_target, folds, ("half_f1",)
    )["half_f1"]
    rows.append({
        "seed": seed,
        "config": RESIDUAL_ONLY,
        "alpha": np.nan,
        "inner_ap": np.nan,
        "hard_weight": np.nan,
        "residual_weight": np.nan,
        **_band_metrics(y_holdout, residual_target, band),
        **score_predictions(y_holdout, residual_target, residual_threshold),
    })

    final_diag = residual_holdout[1]
    correlations = {
        name: float(spearmanr(residual_train, expert_oof[name]).statistic)
        for name in EXPERTS + (ANTI_TEMPLATE_NAME,)
    }
    diagnostics = {
        "seed": seed,
        "e3_spearman": correlations["E3_dx_text"],
        "hard_spearman": correlations[ANTI_TEMPLATE_NAME],
        "max_existing_abs_spearman": max(
            abs(correlations[name]) for name in EXPERTS),
        "final_projection_slope": final_diag.projection_slope,
        "final_calibrator_coefficient": final_diag.calibrator_coefficient,
        "mean_abs_fold_calibrator": float(np.mean([
            abs(diag.calibrator_coefficient) for diag in fold_diagnostics
        ])),
        "final_orthogonality": final_diag.residual_full_correlation,
    }
    return rows, diagnostics


def _formal_report(frame: pd.DataFrame, candidate_label: str) -> None:
    base = frame[frame.config == BASELINE].sort_values("seed")
    candidate = frame[frame.config == candidate_label].sort_values("seed")
    result, deltas = verdict(base, candidate)
    print(f"\n=== H69 formal screen: {candidate_label} ===")
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
        "hard_weight", "residual_weight",
    ]
    print("\n=== strict outer holdout mean +/- std ===")
    print(frame.groupby("config")[metrics].agg(["mean", "std"]).T.to_string(
        float_format=lambda value: f"{value:.5f}"))
    pivot = frame.pivot(index="seed", columns="config")
    for candidate in (HARD_ADD, RESIDUAL_ADD):
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

    print(f"\n=== {RESIDUAL_ADD} - {HARD_ADD} ===")
    for metric in ("auc", "ap", "f1"):
        delta = pivot[(metric, RESIDUAL_ADD)] - pivot[(metric, HARD_ADD)]
        print(
            f"{metric:4s} delta={delta.mean():+.5f} "
            f"wins={int((delta > 1e-12).sum())}/{len(delta)} "
            f"values={np.round(delta.values, 5)}"
        )

    print("\n=== residual diagnostics ===")
    for diag in diagnostics:
        print(
            f"seed={diag['seed']} e3_r={diag['e3_spearman']:+.4f} "
            f"hard_r={diag['hard_spearman']:+.4f} "
            f"max_existing_r={diag['max_existing_abs_spearman']:.4f} "
            f"projection_slope={diag['final_projection_slope']:+.4f} "
            f"calibrator_coef={diag['final_calibrator_coefficient']:+.4f} "
            f"orthogonality={diag['final_orthogonality']:.2e}"
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
        seed_offset = (690 if arguments.seed_offset is None
                       else arguments.seed_offset)
        output = (arguments.output
                  or "exp/oof_contrastive_residual_text_seed690_5rep.csv")
        run_screen(arguments.n_reps, seed_offset, output)
    else:
        seed_offset = (620 if arguments.seed_offset is None
                       else arguments.seed_offset)
        output = (arguments.output
                  or "exp/holdout_contrastive_residual_text_seed620_5rep.csv")
        run_strict(
            arguments.n_reps, seed_offset, arguments.test_size, output)
