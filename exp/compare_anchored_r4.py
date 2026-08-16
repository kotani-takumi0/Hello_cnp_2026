"""H43: exp032 anchored ModernNCA blend のnested二目的評価。

R4はDiscovery全体のbest-threshold F1だけ改善し、APは悪化した。そこで
exp032を最低80%残し、ModernNCAを最大20%だけ混ぜる。各outer-train内の
inner 3-foldで、AP guardrailを通るlambdaのうちthreshold-transfer F1が最大の
ものを選ぶ。lambdaとthresholdはouter-validation labelを見ずに固定する。

  python3 exp/compare_anchored_r4.py --smoke
  python3 exp/compare_anchored_r4.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS, _scores  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402


R4_CACHE = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
OUT_NPZ = Path("exp/_h43_anchored_r4_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_h43_anchored_r4_discovery_seed20260815.json")
LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20)
AP_TOLERANCE = 0.002
INNER_SELECTION_FOLDS = 3
MIN_TRANSFER_F1_GAIN = 0.005
MIN_NONZERO_FOLDS = 3


def _best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    positive = y.astype(bool)[:, None]
    predicted = p[:, None] >= THS[None, :]
    tp = np.sum(predicted & positive, axis=0)
    fp = np.sum(predicted & ~positive, axis=0)
    fn = np.sum(~predicted & positive, axis=0)
    denominator = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denominator,
                       out=np.zeros_like(tp, dtype=float),
                       where=denominator > 0)
    idx = int(np.argmax(scores))
    return float(THS[idx]), float(scores[idx])


def _blend(base: np.ndarray, r4: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * base + lam * r4


def _select_lambda(y: np.ndarray, base: np.ndarray, r4: np.ndarray,
                   seed: int):
    inner = StratifiedKFold(
        INNER_SELECTION_FOLDS, shuffle=True, random_state=seed,
    )
    splits = list(inner.split(np.zeros(len(y)), y))
    rows = []
    for lam in LAMBDAS:
        pred = _blend(base, r4, lam)
        fold_ap, fold_f1, thresholds = [], [], []
        for tr, va in splits:
            threshold, _ = _best_threshold(y[tr], pred[tr])
            thresholds.append(threshold)
            fold_ap.append(float(average_precision_score(y[va], pred[va])))
            fold_f1.append(float(f1_score(y[va], pred[va] >= threshold)))
        rows.append({
            "lambda": lam,
            "mean_ap": float(np.mean(fold_ap)),
            "mean_f1": float(np.mean(fold_f1)),
            "ap_per_fold": fold_ap,
            "f1_per_fold": fold_f1,
            "threshold_per_fold": thresholds,
        })
    base_ap = rows[0]["mean_ap"]
    for row in rows:
        row["ap_delta_vs_base"] = row["mean_ap"] - base_ap
        row["eligible"] = row["mean_ap"] >= base_ap - AP_TOLERANCE
    eligible = [row for row in rows if row["eligible"]]
    # F1最大、次にAP最大、完全同値なら小さいlambdaを選ぶ。
    choice = max(
        eligible,
        key=lambda row: (row["mean_f1"], row["mean_ap"], -row["lambda"]),
    )
    return float(choice["lambda"]), rows


def _error_delta(y, base_label, candidate_label):
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
        "label_disagreement": int((base_label != candidate_label).sum()),
    }


def run(smoke: bool):
    if not R4_CACHE.exists():
        raise FileNotFoundError(f"先にR4 Discovery OOFを作る: {R4_CACHE}")
    data = np.load(R4_CACHE, allow_pickle=True)
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
        train_candidate = _blend(base[tr], r4[tr], lam)
        valid_candidate = _blend(base[va], r4[va], lam)
        candidate_threshold, _ = _best_threshold(y[tr], train_candidate)
        base_threshold, _ = _best_threshold(y[tr], base[tr])
        candidate_oof[va] = valid_candidate
        candidate_label[va] = valid_candidate >= candidate_threshold
        base_label[va] = base[va] >= base_threshold

        row = {
            "fold": fold,
            "lambda": lam,
            "candidate_threshold": candidate_threshold,
            "base_threshold": base_threshold,
            "outer_base_ap": float(average_precision_score(y[va], base[va])),
            "outer_candidate_ap": float(average_precision_score(
                y[va], valid_candidate,
            )),
            "outer_base_f1": float(f1_score(y[va], base_label[va])),
            "outer_candidate_f1": float(f1_score(y[va], candidate_label[va])),
            "inner_grid": inner_rows,
        }
        diagnostics.append(row)
        print(
            f"fold {fold}/{INNER_FOLDS}: lambda={lam:.2f} "
            f"th={candidate_threshold:.3f} "
            f"outer ΔAP={row['outer_candidate_ap'] - row['outer_base_ap']:+.4f} "
            f"ΔF1={row['outer_candidate_f1'] - row['outer_base_f1']:+.4f}"
        )

    if smoke:
        done = np.isfinite(candidate_oof)
        assert done.sum() > 0
        assert ((candidate_oof[done] >= 0) & (candidate_oof[done] <= 1)).all()
        assert (candidate_label[done] >= 0).all()
        print(f"\nSMOKE OK: {done.sum()}件をlambda/threshold非参照foldで評価。")
        return

    assert np.isfinite(candidate_oof).all()
    assert (candidate_label >= 0).all() and (base_label >= 0).all()
    base_score = _scores(y, base)
    candidate_score = _scores(y, candidate_oof)
    ranking_delta = {
        "auc": float(candidate_score["auc"] - base_score["auc"]),
        "ap": float(candidate_score["ap"] - base_score["ap"]),
    }
    transfer = {
        "base_f1": float(f1_score(y, base_label)),
        "candidate_f1": float(f1_score(y, candidate_label)),
    }
    transfer["delta_f1"] = transfer["candidate_f1"] - transfer["base_f1"]
    lambdas = [row["lambda"] for row in diagnostics]
    nonzero_folds = int(sum(lam > 0 for lam in lambdas))
    passed = (
        ranking_delta["ap"] >= -AP_TOLERANCE
        and transfer["delta_f1"] >= MIN_TRANSFER_F1_GAIN
        and nonzero_folds >= MIN_NONZERO_FOLDS
    )
    verdict = "PROMISING" if passed else "REJECT"
    errors = _error_delta(y, base_label, candidate_label)

    table = pd.DataFrame({
        "exp032": {
            "auc": base_score["auc"], "ap": base_score["ap"],
            "global_best_f1_reference": base_score["f1"],
            "transferred_f1": transfer["base_f1"],
        },
        "anchored_R4": {
            "auc": candidate_score["auc"], "ap": candidate_score["ap"],
            "global_best_f1_reference": candidate_score["f1"],
            "transferred_f1": transfer["candidate_f1"],
        },
    }).T
    print("\n=== Nested anchored R4 ===")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"lambda per fold: {lambdas}")
    print(f"ΔAUC {ranking_delta['auc']:+.4f} / ΔAP {ranking_delta['ap']:+.4f} / "
          f"Δtransferred-F1 {transfer['delta_f1']:+.4f}")
    print("Error delta: " + json.dumps(errors, ensure_ascii=False))
    print(f"verdict={verdict}")

    result = {
        "method": "(1-lambda)*exp032 + lambda*raw_ModernNCA",
        "lambda_grid": list(LAMBDAS),
        "ap_tolerance": AP_TOLERANCE,
        "inner_selection_folds": INNER_SELECTION_FOLDS,
        "acceptance": {
            "min_transfer_f1_gain": MIN_TRANSFER_F1_GAIN,
            "min_nonzero_folds": MIN_NONZERO_FOLDS,
        },
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "scores": {
            "exp032": {key: float(value) for key, value in base_score.items()},
            "anchored_R4": {key: float(value)
                            for key, value in candidate_score.items()},
        },
        "ranking_delta": ranking_delta,
        "transferred_threshold": transfer,
        "lambdas": lambdas,
        "nonzero_folds": nonzero_folds,
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
