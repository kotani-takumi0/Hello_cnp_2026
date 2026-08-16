"""R5: Expert不一致によるcross-fitted exp032 logit correction。

R3で「誤りやすさ」は予測できた一方、通常Ridgeのsigned residualは不発だった。そこで
exp032のlogit係数を1に固定し、低自由度の補正項だけをloglossで学ぶ。

    logit(p_new) = logit(p_exp032) + g(disagreement, error_probability)

各外側foldでerror probabilityも内側cross-fitし、同じ行のerror labelを見たin-sample確率を
correction学習へ渡さない。alphaや特徴集合は事前固定し、Discoveryの結果を見た再探索はしない。

  python3 exp/correct_residual.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_residual_predictability import (  # noqa: E402
    CACHE, MODEL_C, base_features, best_threshold, classifier,
)
from ensemble_experts import THS, _scores  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402
from meta_blend import EPS, to_logit  # noqa: E402


ALPHA = 0.1
OUT_JSON = Path("exp/_r5_residual_correction.json")
OUT_NPZ = Path("exp/_r5_residual_correction.npz")


def _loss_grad(theta, base_logit, X, y, alpha):
    w, b = theta[:-1], theta[-1]
    p = expit(base_logit + X @ w + b)
    p = np.clip(p, EPS, 1 - EPS)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + alpha * (w @ w)
    residual = (p - y) / len(y)
    grad = np.concatenate((X.T @ residual + 2 * alpha * w,
                           [residual.sum()]))
    return loss, grad


def fit_correction(base: np.ndarray, X: np.ndarray, y: np.ndarray,
                   alpha: float = ALPHA):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    init = np.zeros(Z.shape[1] + 1)
    result = minimize(
        _loss_grad, init, args=(to_logit(base), Z, y, alpha), jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"correction optimization failed: {result.message}")
    return scaler, result.x[:-1], result.x[-1]


def apply_correction(base: np.ndarray, X: np.ndarray, scaler, w, b):
    return expit(to_logit(base) + scaler.transform(X) @ w + b)


def error_probability_features(base, X, y, tr, va, seed):
    """outer-trainはinner OOF、outer-validはouter-train full fitで誤り確率を作る。"""
    th = best_threshold(y[tr], base[tr])
    err_tr = ((base[tr] >= th).astype(int) != y[tr]).astype(int)
    margin_tr = np.abs(base[tr] - th)[:, None]
    margin_va = np.abs(base[va] - th)[:, None]
    full_tr = np.column_stack((X[tr], margin_tr))
    full_va = np.column_stack((X[va], margin_va))

    q_train = np.zeros(len(tr))
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for i, j in inner.split(full_tr, err_tr):
        q_train[j] = classifier().fit(
            full_tr[i], err_tr[i],
        ).predict_proba(full_tr[j])[:, 1]
    q_valid = classifier().fit(
        full_tr, err_tr,
    ).predict_proba(full_va)[:, 1]
    return (np.column_stack((full_tr, q_train)),
            np.column_stack((full_va, q_valid)), th)


def run():
    z = np.load(CACHE, allow_pickle=True)
    y = z["y"].astype(int)
    names = tuple(z["names"].tolist())
    experts = {n: z[f"oof_{n}"] for n in names}
    base = z["blend"]
    X, feature_names = base_features(base, experts, names)

    corrected = np.zeros(len(y))
    fold_rows = []
    weights = []
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=LOCKBOX_SEED,
    )
    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        Xtr, Xva, th = error_probability_features(
            base, X, y, tr, va, LOCKBOX_SEED + fold * 1000,
        )
        scaler, w, b = fit_correction(base[tr], Xtr, y[tr], ALPHA)
        corrected[va] = apply_correction(base[va], Xva, scaler, w, b)
        weights.append(w)
        fold_rows.append({
            "fold": fold,
            "base_threshold_train": th,
            "base_ap_valid": _scores(y[va], base[va])["ap"],
            "corrected_ap_valid": _scores(y[va], corrected[va])["ap"],
        })

    base_score = _scores(y, base)
    corrected_score = _scores(y, corrected)
    delta = {k: float(corrected_score[k] - base_score[k])
             for k in ("auc", "ap", "f1")}
    avg_w = np.mean(weights, axis=0)
    all_features = feature_names + ["abs_margin_to_train_threshold",
                                    "cross_fitted_error_probability"]

    # 同じglobal OOF最適閾値で誤り遷移を見る。採否はAP/F1差で決める。
    base_label = base >= base_score["th"]
    cand_label = corrected >= corrected_score["th"]
    errors = {
        "base_errors": int((base_label != y).sum()),
        "corrected_errors": int((cand_label != y).sum()),
        "fn_rescued": int(((y == 1) & ~base_label & cand_label).sum()),
        "new_fn": int(((y == 1) & base_label & ~cand_label).sum()),
        "fp_removed": int(((y == 0) & base_label & ~cand_label).sum()),
        "new_fp": int(((y == 0) & ~base_label & cand_label).sum()),
        "label_disagreement": int((base_label != cand_label).sum()),
    }

    print("=== Discovery cross-fitted correction ===")
    print(pd.DataFrame({"exp032": base_score,
                        "corrected": corrected_score}).T.to_string(
                            float_format=lambda x: f"{x:.4f}"))
    print("\n差: " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print("誤り遷移: " + json.dumps(errors, ensure_ascii=False))
    print("\n平均補正係数:")
    for name, value in sorted(zip(all_features, avg_w),
                              key=lambda x: -abs(x[1])):
        print(f"  {name:35s} {value:+.4f}")
    print("\nFold AP:")
    for row in fold_rows:
        d = row["corrected_ap_valid"] - row["base_ap_valid"]
        print(f"  fold {row['fold']}: {row['base_ap_valid']:.4f} -> "
              f"{row['corrected_ap_valid']:.4f} ({d:+.4f})")

    if delta["ap"] >= 0.005 and delta["f1"] > 0:
        verdict = "PASS: 安定性確認候補"
    elif delta["ap"] > 0 and delta["f1"] >= -0.002:
        verdict = "PARK: 微差。新表現追加時だけ再検討"
    else:
        verdict = "REJECT: correction路線を終了"
    print(f"\n判定: {verdict}")

    result = {
        "seed": LOCKBOX_SEED,
        "alpha": ALPHA,
        "feature_names": all_features,
        "base_score": {k: float(v) for k, v in base_score.items()},
        "corrected_score": {k: float(v) for k, v in corrected_score.items()},
        "delta": delta,
        "errors": errors,
        "mean_weights": {k: float(v) for k, v in zip(all_features, avg_w)},
        "folds": fold_rows,
        "verdict": verdict,
        "lockbox_opened": False,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    np.savez_compressed(OUT_NPZ, y=y, base=base, corrected=corrected)
    print(f"保存: {OUT_JSON} / {OUT_NPZ}")


if __name__ == "__main__":
    run()
