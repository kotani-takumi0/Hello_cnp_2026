"""H86: split by p, then run a specialist trained only on the band rows.

Stage 1 is the current blend.  Rows with p<0.1 and p>0.9 keep their score.
Rows in between go to a stage-2 model that has been trained **only on the band
rows of the training set, from the raw features** -- so it can find structure
that the global experts averaged away.

This is not H73.  H73's correction read the eight expert logits, which already
summarise everything the global models know, so it could only reweight them.
It is not H85 either: the class here comes straight from p, nothing is learned
about "which rows are hard".

It is also not ruled out by H85's calibration result.  Calibration says
E[y | p] = p inside the band; it does not say E[y | p, x] = p.  A specialist
that re-ranks within the band is still allowed by that finding.

  logit_final = logit(p)                       outside the band
              = logit(p) + λ·(logit(p2) - logit(p̄))   inside the band

λ is chosen inside the fold's training part by cross-fit AP **restricted to the
band**, over a grid that includes 0.  So the arm can decline to use the
specialist, and a λ that stays at 0 is itself the answer.

Two specialists are compared, because n is the whole problem: a fold's training
band holds roughly 180 rows.

  band_lr    L2 logistic regression on the anchor columns
  band_lgbm  a very small LightGBM

Measured on the production fold structure (blend_oof's outer 5-fold), not the
A/B holdout -- the band is defined by the blend's own output, which is the shape
that A/B flatters (meta_native_and_alpha_selection_20260829.md section 4.4).

Run:
  .venv/bin/python exp/compare_band_specialist.py --seeds 0 1 2 3 4 5 6 7 8 9 42
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import _inputs  # noqa: E402
from compare_meta_native import _oof_logit  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from harness import LGB_THREADS  # noqa: E402
from meta_blend import EPS, apply_meta, fit_meta, select_alpha_1se  # noqa: E402

BAND_LO, BAND_HI = 0.1, 0.9
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0)
BASELINE = "meta_only"
SPECIALISTS = ("band_lr", "band_lgbm")
SPECIALIST_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=4,
    min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=LGB_THREADS,
)
SPECIALIST_ROUNDS = 120


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _numeric(frame):
    """帯専門モデル用。カテゴリは one-hot ではなく落とす（n が小さすぎる）。"""
    return frame.select_dtypes(include=[np.number])


def _fit_specialist(kind, X_fit, y_fit, X_apply, seed):
    if kind == "band_lr":
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.05, max_iter=5000, random_state=seed))
        model.fit(X_fit, y_fit)
        return model.predict_proba(X_apply)[:, 1]
    model = lgb.train(
        {**SPECIALIST_PARAMS, "seed": seed},
        lgb.Dataset(X_fit, y_fit), num_boost_round=SPECIALIST_ROUNDS)
    return model.predict(X_apply)


def _select_lambda(kind, X_band, y_band, base_logit_band, seed):
    """fold train の帯行の中だけで λ を選ぶ。基準は帯内AP。"""
    if len(y_band) < 40 or len(np.unique(y_band)) < 2:
        return 0.0, np.nan
    oof = np.zeros(len(y_band))
    for tr, va in StratifiedKFold(
            4, shuffle=True, random_state=seed).split(X_band, y_band):
        if len(np.unique(y_band[tr])) < 2:
            oof[va] = y_band.mean()
            continue
        oof[va] = _fit_specialist(
            kind, X_band.iloc[tr], y_band[tr], X_band.iloc[va], seed)
    centre = _logit(np.clip(y_band.mean(), EPS, 1 - EPS))
    solo_ap = average_precision_score(y_band, oof)
    best, best_ap = 0.0, -np.inf
    for lam in LAMBDA_GRID:
        score = base_logit_band + lam * (_logit(oof) - centre)
        ap = average_precision_score(y_band, score)
        if ap > best_ap:
            best, best_ap = lam, ap
    return best, solo_ap


def _one_seed(y, Z, X, columns, seed):
    numeric_X = _numeric(X[columns])
    out = {BASELINE: np.zeros(len(y))}
    for kind in SPECIALISTS:
        out[kind] = np.zeros(len(y))
    detail = []

    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        alpha = select_alpha_1se(Z[tr], y[tr], seed)
        weight, intercept = fit_meta(Z[tr], y[tr], alpha=alpha)
        base_va_logit = Z[va] @ weight + intercept
        out[BASELINE][va] = expit(base_va_logit)

        inner_logit = _oof_logit(Z[tr], y[tr], alpha, seed)
        inner_p = expit(inner_logit)
        band_tr = (inner_p > BAND_LO) & (inner_p < BAND_HI)
        band_va = (out[BASELINE][va] > BAND_LO) & (out[BASELINE][va] < BAND_HI)

        X_band = numeric_X.iloc[tr[band_tr]]
        y_band = y[tr][band_tr]
        centre = _logit(np.clip(y_band.mean(), EPS, 1 - EPS)) if len(y_band) else 0.0

        row = {"seed": seed, "band_train_n": int(band_tr.sum()),
               "band_valid_n": int(band_va.sum())}
        for kind in SPECIALISTS:
            lam, solo_ap = _select_lambda(
                kind, X_band, y_band, inner_logit[band_tr], seed)
            score = base_va_logit.copy()
            if lam > 0 and band_va.sum():
                p2 = _fit_specialist(
                    kind, X_band, y_band, numeric_X.iloc[va[band_va]], seed)
                score[band_va] += lam * (_logit(p2) - centre)
            out[kind][va] = expit(score)
            row[f"lambda[{kind}]"] = lam
            row[f"帯内単体AP[{kind}]"] = solo_ap
        row["帯の実購入率"] = float(y_band.mean()) if len(y_band) else np.nan
        detail.append(row)
    return out, detail


def main(seeds, config):
    X, y_ref, _, _, _, columns = _inputs()
    anchor = columns[0]
    rows, details = [], []
    for seed in seeds:
        y, Z, _ = _load(CONFIGS[config], seed)
        assert np.array_equal(y, y_ref)
        out, detail = _one_seed(y, Z, X, anchor, seed)
        details.extend(detail)
        for name, p in out.items():
            rows.append({"seed": seed, "arm": name,
                         "auc": roc_auc_score(y, p),
                         "ap": average_precision_score(y, p),
                         "f1": _best_f1(y, p)})
    result = pd.DataFrame(rows)
    frame = pd.DataFrame(details)

    print("=== 帯専門モデルの内部量（55 fold の平均）===")
    print(frame.mean(numeric_only=True).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\n  λ が選ばれた回数（0 なら専門モデルを使わないという判断）")
    for kind in SPECIALISTS:
        counts = frame[f"lambda[{kind}]"].value_counts().sort_index()
        print(f"    {kind:11s} " +
              "  ".join(f"λ={k:g}: {v}fold" for k, v in counts.items()))

    print("\n=== 本番と同じ外側5fold での成績 ===")
    print(result.groupby("arm")[["auc", "ap", "f1"]].mean().reindex(
        [BASELINE, *SPECIALISTS]).to_string(float_format=lambda v: f"{v:.5f}"))
    base = result[result.arm == BASELINE].sort_values("seed").reset_index(drop=True)
    for kind in SPECIALISTS:
        cand = result[result.arm == kind].sort_values("seed").reset_index(drop=True)
        line = [f"  {kind:11s}"]
        for metric in ("auc", "ap", "f1"):
            d = cand[metric].values - base[metric].values
            line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
        decision, _ = verdict(base, cand)
        print("  ".join(line) + f"   {decision}")
    result.to_csv("exp/band_specialist.csv", index=False)
    frame.to_csv("exp/band_specialist_folds.csv", index=False)
    print("\nsaved: exp/band_specialist.csv, exp/band_specialist_folds.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 42])
    args = parser.parse_args()
    main(args.seeds, args.config)
