"""H85: predict the confidence class itself, instead of reading it off p.

The ambiguous band is currently defined by the blend's own probability:
p<0.1 is confident-negative, p>0.9 confident-positive, the rest ambiguous.
The idea under test is to learn that three-way class directly from the raw
features, so that "which companies are hard to judge" stops being a restatement
of the current model's uncertainty.

The whole question is whether that is circular.  A three-class model trained on
labels derived from p can only be useful if the features let it say something
about the class that p does not already say.  So the script reports, in order:

  1. how well the class model predicts the class at all (it should be good --
     features predict p, which defines the class, so this is nearly tautological)
  2. **whether its output adds anything to the blend** -- the three class
     probabilities are appended to the meta input and the paired delta in AP is
     measured.  This is the decisive number; step 1 can look excellent while
     step 2 is zero.
  3. whether the band is conditionally miscalibrated, which is the one mechanism
     by which a per-class decision rule could beat one global threshold.  If the
     blend is calibrated inside the band, no gate on class membership can help,
     because the F1-optimal rule on calibrated probabilities is a single cut.

Measured on the production fold structure (blend_oof's outer 5-fold over all 742
rows), not on the A/B holdout -- a stage that learns from the blend's output is
exactly the shape that the A/B split flatters
(documents/meta_native_and_alpha_selection_20260829.md section 4.4).

Run:
  .venv/bin/python exp/compare_three_class_gate.py --seeds 0 1 2 3 4 5 6 7 8 9 42
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import _inputs  # noqa: E402
from compare_meta_native import _oof_logit  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from harness import LGB_THREADS  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha_1se  # noqa: E402

BAND_LO, BAND_HI = 0.1, 0.9
CLASS_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7,
    min_child_samples=25, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, verbosity=-1, num_threads=LGB_THREADS,
)
CLASS_ROUNDS = 200
BASELINE = "meta_only"
CANDIDATE = "meta_plus_class"


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _confidence_class(p):
    return np.where(p < BAND_LO, 0, np.where(p > BAND_HI, 2, 1))


def _one_seed(y, Z, X, columns, seed):
    """本番と同じ外側5fold。fold trainの中だけでクラス定義もモデルも作る。"""
    base = np.zeros(len(y))
    cand = np.zeros(len(y))
    class_probability = np.zeros((len(y), 3))
    class_truth = np.zeros(len(y), dtype=int)

    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        alpha = select_alpha_1se(Z[tr], y[tr], seed)
        weight, intercept = fit_meta(Z[tr], y[tr], alpha=alpha)
        base[va] = apply_meta(Z[va], weight, intercept)

        # クラスは fold train の中の交差適合 p から定義する（vaは見ない）
        inner_p = expit(_oof_logit(Z[tr], y[tr], alpha, seed))
        label = _confidence_class(inner_p)
        class_truth[va] = _confidence_class(base[va])

        if len(np.unique(label)) < 3:
            class_probability[va] = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            model = lgb.train(
                {**CLASS_PARAMS, "seed": seed},
                lgb.Dataset(X.iloc[tr][columns], label),
                num_boost_round=CLASS_ROUNDS)
            class_probability[va] = model.predict(X.iloc[va][columns])
            train_probability = model.predict(X.iloc[tr][columns])

            augmented_tr = np.column_stack([Z[tr], train_probability])
            augmented_va = np.column_stack([Z[va], class_probability[va]])
            alpha_c = select_alpha_1se(augmented_tr, y[tr], seed)
            w_c, b_c = fit_meta(augmented_tr, y[tr], alpha=alpha_c)
            cand[va] = apply_meta(augmented_va, w_c, b_c)
    return base, cand, class_probability, class_truth


def _band_calibration(y, p):
    """帯の中だけで、予測確率と実測購入率のズレを見る。"""
    band = (p > BAND_LO) & (p < BAND_HI)
    if band.sum() < 30:
        return None
    order = np.argsort(p[band])
    chunks = np.array_split(order, 4)
    rows = []
    for i, idx in enumerate(chunks):
        rows.append({
            "四分位": i + 1, "n": len(idx),
            "予測p平均": float(p[band][idx].mean()),
            "実測購入率": float(y[band][idx].mean()),
        })
    frame = pd.DataFrame(rows)
    frame["差"] = frame["実測購入率"] - frame["予測p平均"]
    # 各ビンの標準誤差（実測率の二項分布）
    frame["se"] = np.sqrt(
        frame["実測購入率"] * (1 - frame["実測購入率"]) / frame["n"])
    frame["z"] = frame["差"] / frame["se"].replace(0, np.nan)
    return frame


def main(seeds, config):
    X, y_ref, _, _, _, columns = _inputs()
    feature_columns = columns[0]
    rows, class_rows, calibration = [], [], []
    for seed in seeds:
        y, Z, _ = _load(CONFIGS[config], seed)
        assert np.array_equal(y, y_ref)
        base, cand, class_p, class_y = _one_seed(
            y, Z, X, feature_columns, seed)
        for name, p in ((BASELINE, base), (CANDIDATE, cand)):
            rows.append({"seed": seed, "arm": name,
                         "auc": roc_auc_score(y, p),
                         "ap": average_precision_score(y, p),
                         "f1": _best_f1(y, p)})
        # クラス予測そのものの当たり具合（1対他のAUC）
        entry = {"seed": seed,
                 "3クラス正答率": float((class_p.argmax(1) == class_y).mean())}
        for k, label in enumerate(("確信-非購入", "曖昧", "確信-購入")):
            if len(np.unique(class_y == k)) > 1:
                entry[f"AUC[{label}]"] = roc_auc_score(class_y == k, class_p[:, k])
        class_rows.append(entry)
        frame = _band_calibration(y, base)
        if frame is not None:
            frame["seed"] = seed
            calibration.append(frame)

    result = pd.DataFrame(rows)
    print("=== 1. クラス予測そのものの当たり具合（ほぼ自明に高いはず）===")
    print(pd.DataFrame(class_rows).mean(numeric_only=True).to_string(
        float_format=lambda v: f"{v:.4f}"))

    print("\n=== 2. 決定的な数字: 合成に足して動くか ===")
    print(result.groupby("arm")[["auc", "ap", "f1"]].mean().to_string(
        float_format=lambda v: f"{v:.5f}"))
    b = result[result.arm == BASELINE].sort_values("seed").reset_index(drop=True)
    c = result[result.arm == CANDIDATE].sort_values("seed").reset_index(drop=True)
    line = ["  ペア差 "]
    for metric in ("auc", "ap", "f1"):
        d = c[metric].values - b[metric].values
        line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
    decision, _ = verdict(b, c)
    print("  ".join(line) + f"   {decision}")

    print("\n=== 3. 帯の中の較正（seed平均、ズレが無ければクラス別閾値も無意味）===")
    merged = pd.concat(calibration).groupby("四分位").mean(numeric_only=True)
    print(merged[["n", "予測p平均", "実測購入率", "差", "z"]].to_string(
        float_format=lambda v: f"{v:.4f}"))
    result.to_csv("exp/three_class_gate.csv", index=False)
    print("\nsaved: exp/three_class_gate.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 42])
    args = parser.parse_args()
    main(args.seeds, args.config)
