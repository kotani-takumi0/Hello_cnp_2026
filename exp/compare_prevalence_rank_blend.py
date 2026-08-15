"""H44: exp032 / ModernNCA の陽性数制約付きrank blendをnested評価する。

H43はthreshold移動によりtestの予測陽性が13件増え、Publicに現れた4件が
すべてFPだった。今回は各validation foldでexp032の転送thresholdが作る陽性数
Kをそのまま予算とし、候補はrank blendの上位K件だけを陽性にする。

lambdaとKの基準になるexp032 thresholdはouter-validation labelを見ずに決める。

  python3 exp/compare_prevalence_rank_blend.py --smoke
  python3 exp/compare_prevalence_rank_blend.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import _scores  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402
from compare_anchored_r4 import _best_threshold  # noqa: E402


R4_CACHE = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
OUT_NPZ = Path("exp/_h44_prevalence_rank_blend_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_h44_prevalence_rank_blend_discovery_seed20260815.json")
LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
AP_TOLERANCE = 0.002
INNER_SELECTION_FOLDS = 3
MIN_TRANSFER_F1_GAIN = 0.005
MIN_NONZERO_FOLDS = 3
MAX_SINGLE_FOLD_F1_DROP = 0.02


def _rank_score(score: np.ndarray) -> np.ndarray:
    """スコア尺度を消し、同順位には同じ[0, 1] percentileを与える。"""
    score = np.asarray(score, dtype=float)
    return rankdata(score, method="average") / len(score)


def _blend_rank(base: np.ndarray, r4: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * _rank_score(base) + lam * _rank_score(r4)


def _top_k_label(score: np.ndarray, k: int) -> np.ndarray:
    """同点時もindex順で決定的に、正確にk件だけ1を返す。"""
    if not 0 <= k <= len(score):
        raise ValueError(f"invalid k={k} for n={len(score)}")
    label = np.zeros(len(score), dtype=int)
    if k:
        order = np.argsort(np.asarray(score), kind="mergesort")
        label[order[-k:]] = 1
    return label


def _select_lambda(y: np.ndarray, base: np.ndarray, r4: np.ndarray,
                   seed: int):
    inner = StratifiedKFold(
        INNER_SELECTION_FOLDS, shuffle=True, random_state=seed,
    )
    splits = list(inner.split(np.zeros(len(y)), y))
    rows = []
    for lam in LAMBDAS:
        fold_ap, fold_f1, fold_base_f1, positive_budgets = [], [], [], []
        for tr, va in splits:
            base_threshold, _ = _best_threshold(y[tr], base[tr])
            base_label = (base[va] >= base_threshold).astype(int)
            k = int(base_label.sum())
            candidate_score = _blend_rank(base[va], r4[va], lam)
            candidate_label = _top_k_label(candidate_score, k)
            fold_ap.append(float(average_precision_score(
                y[va], candidate_score,
            )))
            fold_f1.append(float(f1_score(y[va], candidate_label)))
            fold_base_f1.append(float(f1_score(y[va], base_label)))
            positive_budgets.append(k)
        rows.append({
            "lambda": lam,
            "mean_ap": float(np.mean(fold_ap)),
            "mean_f1": float(np.mean(fold_f1)),
            "mean_base_f1": float(np.mean(fold_base_f1)),
            "ap_per_fold": fold_ap,
            "f1_per_fold": fold_f1,
            "base_f1_per_fold": fold_base_f1,
            "positive_budgets": positive_budgets,
        })
    base_ap = rows[0]["mean_ap"]
    for row in rows:
        row["ap_delta_vs_base"] = row["mean_ap"] - base_ap
        row["f1_delta_vs_base"] = row["mean_f1"] - row["mean_base_f1"]
        row["eligible"] = row["mean_ap"] >= base_ap - AP_TOLERANCE
    eligible = [row for row in rows if row["eligible"]]
    choice = max(
        eligible,
        key=lambda row: (row["mean_f1"], row["mean_ap"], -row["lambda"]),
    )
    return float(choice["lambda"]), rows


def _error_delta(y: np.ndarray, base_label: np.ndarray,
                 candidate_label: np.ndarray):
    return {
        "base_errors": int((base_label != y).sum()),
        "candidate_errors": int((candidate_label != y).sum()),
        "fn_rescued": int(((y == 1) & (base_label == 0) &
                           (candidate_label == 1)).sum()),
        "new_fn": int(((y == 1) & (base_label == 1) &
                       (candidate_label == 0)).sum()),
        "fp_removed": int(((y == 0) & (base_label == 1) &
                           (candidate_label == 0)).sum()),
        "new_fp": int(((y == 0) & (base_label == 0) &
                       (candidate_label == 1)).sum()),
        "zero_to_one": int(((base_label == 0) &
                            (candidate_label == 1)).sum()),
        "one_to_zero": int(((base_label == 1) &
                            (candidate_label == 0)).sum()),
        "label_disagreement": int((base_label != candidate_label).sum()),
    }


def run(smoke: bool):
    if not R4_CACHE.exists():
        raise FileNotFoundError(f"先にR4 Discovery OOFを作る: {R4_CACHE}")
    data = np.load(R4_CACHE)
    y = data["y"].astype(int)
    base = np.asarray(data["base_blend"], dtype=float)
    r4 = np.asarray(data["modern_nca_oof"], dtype=float)
    assert y.shape == base.shape == r4.shape
    assert np.isfinite(base).all() and np.isfinite(r4).all()

    folds = StratifiedKFold(
        INNER_FOLDS, shuffle=True, random_state=LOCKBOX_SEED,
    )
    candidate_oof = np.full(len(y), np.nan, dtype=float)
    candidate_label = np.full(len(y), -1, dtype=int)
    base_label = np.full(len(y), -1, dtype=int)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(np.zeros(len(y)), y), 1):
        if smoke and fold > 1:
            break
        lam, inner_rows = _select_lambda(
            y[tr], base[tr], r4[tr], LOCKBOX_SEED + fold * 10000,
        )
        base_threshold, _ = _best_threshold(y[tr], base[tr])
        outer_base_label = (base[va] >= base_threshold).astype(int)
        k = int(outer_base_label.sum())
        outer_candidate_score = _blend_rank(base[va], r4[va], lam)
        outer_candidate_label = _top_k_label(outer_candidate_score, k)

        base_label[va] = outer_base_label
        candidate_oof[va] = outer_candidate_score
        candidate_label[va] = outer_candidate_label
        assert int(candidate_label[va].sum()) == int(base_label[va].sum())

        row = {
            "fold": fold,
            "lambda": lam,
            "base_threshold": base_threshold,
            "positive_budget": k,
            "outer_base_ap": float(average_precision_score(y[va], base[va])),
            "outer_candidate_ap": float(average_precision_score(
                y[va], outer_candidate_score,
            )),
            "outer_base_f1": float(f1_score(y[va], base_label[va])),
            "outer_candidate_f1": float(f1_score(
                y[va], candidate_label[va],
            )),
            "swaps_each_direction": int(
                ((base_label[va] == 0) & (candidate_label[va] == 1)).sum()
            ),
            "inner_grid": inner_rows,
        }
        diagnostics.append(row)
        print(
            f"fold {fold}/{INNER_FOLDS}: lambda={lam:.2f} K={k} "
            f"swaps={row['swaps_each_direction']} "
            f"outer ΔAP={row['outer_candidate_ap'] - row['outer_base_ap']:+.4f} "
            f"ΔF1={row['outer_candidate_f1'] - row['outer_base_f1']:+.4f}"
        )

    if smoke:
        done = np.isfinite(candidate_oof)
        assert done.sum() > 0
        assert (candidate_label[done] >= 0).all()
        assert candidate_label[done].sum() == base_label[done].sum()
        print(f"\nSMOKE OK: {done.sum()}件を同一陽性数でfold外評価。")
        return

    assert np.isfinite(candidate_oof).all()
    assert (candidate_label >= 0).all() and (base_label >= 0).all()
    assert int(candidate_label.sum()) == int(base_label.sum())
    base_score = _scores(y, base)
    candidate_score = _scores(y, candidate_oof)
    ranking_delta = {
        "auc": float(roc_auc_score(y, candidate_oof)
                     - roc_auc_score(y, base)),
        "ap": float(average_precision_score(y, candidate_oof)
                    - average_precision_score(y, base)),
    }
    transfer = {
        "base_f1": float(f1_score(y, base_label)),
        "candidate_f1": float(f1_score(y, candidate_label)),
    }
    transfer["delta_f1"] = transfer["candidate_f1"] - transfer["base_f1"]
    fold_f1_deltas = [
        row["outer_candidate_f1"] - row["outer_base_f1"]
        for row in diagnostics
    ]
    lambdas = [row["lambda"] for row in diagnostics]
    nonzero_folds = int(sum(lam > 0 for lam in lambdas))
    passed = (
        ranking_delta["ap"] >= -AP_TOLERANCE
        and transfer["delta_f1"] >= MIN_TRANSFER_F1_GAIN
        and nonzero_folds >= MIN_NONZERO_FOLDS
        and min(fold_f1_deltas) >= -MAX_SINGLE_FOLD_F1_DROP
    )
    verdict = "PROMISING" if passed else "REJECT"
    errors = _error_delta(y, base_label, candidate_label)
    assert errors["zero_to_one"] == errors["one_to_zero"]

    table = pd.DataFrame({
        "exp032": {
            "auc": base_score["auc"], "ap": base_score["ap"],
            "global_best_f1_reference": base_score["f1"],
            "budgeted_f1": transfer["base_f1"],
        },
        "fixed_K_rank_blend": {
            "auc": candidate_score["auc"], "ap": candidate_score["ap"],
            "global_best_f1_reference": candidate_score["f1"],
            "budgeted_f1": transfer["candidate_f1"],
        },
    }).T
    print("\n=== Nested prevalence-constrained rank blend ===")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"lambda per fold: {lambdas}")
    print(f"positive budget: base={int(base_label.sum())} "
          f"candidate={int(candidate_label.sum())}")
    print(f"ΔAUC {ranking_delta['auc']:+.4f} / ΔAP {ranking_delta['ap']:+.4f} / "
          f"Δbudgeted-F1 {transfer['delta_f1']:+.4f}")
    print("fold ΔF1: " + " / ".join(f"{v:+.4f}" for v in fold_f1_deltas))
    print("Error delta: " + json.dumps(errors, ensure_ascii=False))
    print(f"verdict={verdict}")

    result = {
        "method": "(1-lambda)*rank(exp032) + lambda*rank(raw_ModernNCA)",
        "prediction_rule": (
            "candidate uses exactly K positives, where K is the number of "
            "base validation scores above the outer-train base threshold"
        ),
        "lambda_grid": list(LAMBDAS),
        "ap_tolerance": AP_TOLERANCE,
        "inner_selection_folds": INNER_SELECTION_FOLDS,
        "acceptance": {
            "min_budgeted_f1_gain": MIN_TRANSFER_F1_GAIN,
            "min_nonzero_folds": MIN_NONZERO_FOLDS,
            "max_single_fold_f1_drop": MAX_SINGLE_FOLD_F1_DROP,
        },
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "scores": {
            "exp032": {key: float(value) for key, value in base_score.items()},
            "fixed_K_rank_blend": {
                key: float(value) for key, value in candidate_score.items()
            },
        },
        "ranking_delta": ranking_delta,
        "budgeted_f1": transfer,
        "lambdas": lambdas,
        "nonzero_folds": nonzero_folds,
        "positive_budget": int(base_label.sum()),
        "fold_f1_deltas": fold_f1_deltas,
        "errors": errors,
        "fold_diagnostics": diagnostics,
        "verdict": verdict,
        "lockbox_opened": False,
    }
    np.savez_compressed(
        OUT_NPZ, y=y, exp032=base, candidate=candidate_oof,
        exp032_label=base_label, candidate_label=candidate_label,
        lambdas=np.asarray(lambdas),
    )
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"保存: {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
