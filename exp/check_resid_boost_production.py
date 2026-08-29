"""H82 on the production path: residual boosting inside blend_oof's outer folds.

The strict holdout says a LightGBM trained on the fixed-alpha blend logit as an
offset adds AP with 20/20 sign consistency.  That holdout fits the meta on 482
rows; production fits it on 594 and predicts 800 test rows, so the effect has to
be re-measured on the real fold sizes before anything is wired in.

Per seed this reproduces ``blend_oof`` exactly -- same outer 5-fold, same alpha
selection inside each fold's training part -- and adds one stage:

  1. fit the meta on the outer fold's training rows
  2. cross-fit that same meta inside those rows to get an honest offset
  3. fit the residual model on the training rows with that offset
  4. add its raw score to the meta logit for the validation rows and for test

No outer-validation label ever reaches step 1, 2, or 3, so the resulting OOF is
comparable to the auto/1se columns from ``diagnose_meta_alpha.py``.

Result (11 seeds): the boost is -0.0063 AP (1/11) against plain 1se and -0.0023
AUC (0/11), and it is worse than its own row-shuffled placebo (-0.0037), so
there is no learned signal at production fold sizes -- the holdout gain came
from that protocol, not from the features.

Run:
  .venv/bin/python exp/check_resid_boost_production.py --seeds 0 1 2 3 4 42
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import _inputs  # noqa: E402
from compare_meta_native import _oof_logit, _residual_model  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS, build_features  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha, select_alpha_1se  # noqa: E402

RULES = ("auto", "1se", "1se_boost", "1se_boost_placebo")


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _blend_with_boost(y, Z, Zte, X, X_test, resid_cols, seed):
    """1seedぶん。4通りの合成OOFとtest確率をまとめて返す。"""
    out = {r: np.zeros(len(y)) for r in RULES}
    test = {r: [] for r in RULES}
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        a_auto = select_alpha(Z[tr], y[tr], seed=seed)[0]
        w_a, b_a = fit_meta(Z[tr], y[tr], alpha=a_auto)
        out["auto"][va] = apply_meta(Z[va], w_a, b_a)
        test["auto"].append(apply_meta(Zte, w_a, b_a))

        a_1se = select_alpha_1se(Z[tr], y[tr], seed)
        w_s, b_s = fit_meta(Z[tr], y[tr], alpha=a_1se)
        logit_va = Z[va] @ w_s + b_s
        logit_te = Zte @ w_s + b_s
        out["1se"][va] = expit(logit_va)
        test["1se"].append(expit(logit_te))

        offset = _oof_logit(Z[tr], y[tr], a_1se, seed)
        X_tr = X.iloc[tr][resid_cols]
        model = _residual_model(X_tr, y[tr], offset, seed)
        out["1se_boost"][va] = expit(
            logit_va + model.predict(X.iloc[va][resid_cols], raw_score=True))
        test["1se_boost"].append(expit(
            logit_te + model.predict(X_test[resid_cols], raw_score=True)))

        shuffled = X_tr.iloc[
            np.random.default_rng(seed + 991).permutation(len(X_tr))
        ].reset_index(drop=True)
        placebo = _residual_model(shuffled, y[tr], offset, seed)
        out["1se_boost_placebo"][va] = expit(
            logit_va + placebo.predict(X.iloc[va][resid_cols], raw_score=True))
        test["1se_boost_placebo"].append(expit(
            logit_te + placebo.predict(X_test[resid_cols], raw_score=True)))
    return out, {r: np.mean(v, axis=0) for r, v in test.items()}


def main(seeds, config):
    X, y_ref, _, _, _, columns = _inputs()
    resid_cols = columns[0]
    _, _, X_test = build_features(
        pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv"),
        cross=True, llm_makebuy=True, h2_restore=True, h7_restore=True)

    rows, flips = [], []
    for seed in seeds:
        y, Z, Zte = _load(CONFIGS[config], seed)
        assert np.array_equal(y, y_ref), "キャッシュと再構築でyが一致しない"
        blend, test_probability = _blend_with_boost(
            y, Z, Zte, X, X_test, resid_cols, seed)
        labels = {}
        for rule in RULES:
            threshold = THS[int(np.argmax(
                [f1_score(y, (blend[rule] >= t).astype(int)) for t in THS]))]
            labels[rule] = (test_probability[rule] >= threshold).astype(int)
            rows.append({
                "seed": seed, "rule": rule,
                "auc": roc_auc_score(y, blend[rule]),
                "ap": average_precision_score(y, blend[rule]),
                "f1": _best_f1(y, blend[rule]),
                "test_positives": int(labels[rule].sum()),
            })
        flips.append({"seed": seed,
                      **{f"vs_{r}": int((labels["auto"] != labels[r]).sum())
                         for r in RULES if r != "auto"}})

    frame = pd.DataFrame(rows)
    print("=== seed 平均（train OOF、本番と同じ外側5fold）===")
    print(frame.groupby("rule")[["auc", "ap", "f1"]].mean().reindex(
        RULES).to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n=== ペア差 ===")
    for base_rule, arm in (("auto", "1se"), ("auto", "1se_boost"),
                           ("1se", "1se_boost"),
                           ("1se", "1se_boost_placebo")):
        b = frame[frame["rule"] == base_rule].sort_values("seed")
        c = frame[frame["rule"] == arm].sort_values("seed")
        line = [f"  {arm:18s} - {base_rule:10s}"]
        for metric in ("auc", "ap", "f1"):
            delta = c[metric].values - b[metric].values
            line.append(f"{metric} {delta.mean():+.5f} "
                        f"({int((delta > 0).sum())}/{len(delta)})")
        decision, _ = verdict(b.reset_index(drop=True), c.reset_index(drop=True))
        print("  ".join(line) + f"   {decision}")

    print("\n=== auto の提出ラベルから何行動くか（test 800行）===")
    print(pd.DataFrame(flips).set_index("seed").to_string())
    frame.to_csv("exp/resid_boost_production.csv", index=False)
    print("\nsaved: exp/resid_boost_production.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 42])
    args = parser.parse_args()
    main(args.seeds, args.config)
