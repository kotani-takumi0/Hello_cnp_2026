"""H87: split the *information sources* across stages, not just the rows.

The user's proposal: judge coarsely with one feature group, then resolve the
ambiguous cases with a different one -- finance first, survey second.

This is not H86.  There the stage-2 model read the same anchor columns the
stage-1 model already used, so it could only restate them.  Here stage 1 is
deliberately blind to the survey, and the survey is what breaks the tie.

Two facts make it worth testing:

  * E2_survey holds the most unique information of the eight experts
    (leave-one-out ΔAP +0.0235) and is nearly orthogonal to the rest
    (mean |r| 0.129) -- expert_architecture_audit_20260829.md
  * inside the ambiguous band the only raw column that clears a 95% CI is
    アンケート８ -- uncertain_band_profile_20260828.md

And the structural argument: the production meta is **additive** in the expert
logits.  "The survey matters only when the finance side is inconclusive" is a
conditional statement that an additive combiner cannot express, however it is
weighted.  So this is a model-class difference, not a reweighting.

Arms (all reuse the cached expert predictions, so no expert is refit):

  full_meta              current 8 experts, additive          <- baseline
  stage1_only            6 experts with the survey removed    <- reference
  cascade                stage1 + survey applied ONLY in stage1's band
  full_plus_survey_band  full 8 + an extra survey term in the band

`cascade` vs `full_meta` is the head-to-head the proposal asks for: the same
information, used conditionally instead of additively.  `stage1_only` shows what
removing the survey costs, so the cascade's job is to win that back and more.

λ comes from a grid containing 0, chosen inside each fold's training part by AP
restricted to the band, so an arm can decline to use its stage 2.

Measured on the production outer 5-fold (blend_oof), not the A/B holdout.

Run:
  .venv/bin/python exp/compare_survey_cascade.py --seeds 0 1 2 3 4 5 6 7 8 9 42
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
from compare_meta_native import _oof_logit  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, EXPERTS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from meta_blend import EPS, apply_meta, fit_meta, select_alpha_1se  # noqa: E402

BAND_LO, BAND_HI = 0.1, 0.9
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 1.5)
# 1段目からアンケート由来を外す。E7_cross の入力は不満2/不満4 × 財務なので
# アンケート由来であり、外さないと「アンケートを見ていない1段目」にならない。
SURVEY_EXPERTS = ("E2_survey", "E7_cross")
STAGE1 = tuple(n for n in EXPERTS if n not in SURVEY_EXPERTS)
FULL = "full_meta"
STAGE1_ONLY = "stage1_only"
CASCADE = "cascade"
FULL_PLUS = "full_plus_survey_band"
SPLIT = "survey_weight_split"
ARMS = (FULL, STAGE1_ONLY, CASCADE, FULL_PLUS, SPLIT)


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _fit_meta_on(Z, y, seed):
    alpha = select_alpha_1se(Z, y, seed)
    weight, intercept = fit_meta(Z, y, alpha=alpha)
    return alpha, weight, intercept


def _choose_lambda(base_logit, survey_logit, y, band, centre):
    """帯の中のAPで λ を選ぶ。グリッドに 0 を含むので不使用も選べる。"""
    if band.sum() < 30 or len(np.unique(y[band])) < 2:
        return 0.0
    best, best_ap = 0.0, -np.inf
    for lam in LAMBDA_GRID:
        score = base_logit[band] + lam * (survey_logit[band] - centre)
        ap = average_precision_score(y[band], score)
        if ap > best_ap:
            best, best_ap = lam, ap
    return best


def _one_seed(y, Z_all, seed):
    index = {name: i for i, name in enumerate(EXPERTS)}
    stage1_cols = [index[n] for n in STAGE1]
    survey_col = index["E2_survey"]

    out = {arm: np.zeros(len(y)) for arm in ARMS}
    detail = []

    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z_all, y):
        # --- 全部入りの現行メタ ---------------------------------------------
        alpha_f, w_f, b_f = _fit_meta_on(Z_all[tr], y[tr], seed)
        full_va = Z_all[va] @ w_f + b_f
        out[FULL][va] = expit(full_va)
        full_inner = _oof_logit(Z_all[tr], y[tr], alpha_f, seed)

        # --- 1段目: アンケート由来を外した6本 --------------------------------
        Z1 = Z_all[:, stage1_cols]
        alpha_1, w_1, b_1 = _fit_meta_on(Z1[tr], y[tr], seed)
        stage1_va = Z1[va] @ w_1 + b_1
        out[STAGE1_ONLY][va] = expit(stage1_va)
        stage1_inner = _oof_logit(Z1[tr], y[tr], alpha_1, seed)

        survey_tr = Z_all[tr][:, survey_col]
        survey_va = Z_all[va][:, survey_col]
        centre = float(np.median(survey_tr))

        # --- cascade: 1段目の帯だけ、アンケートで決着させる ------------------
        band_tr = ((expit(stage1_inner) > BAND_LO)
                   & (expit(stage1_inner) < BAND_HI))
        lam_c = _choose_lambda(stage1_inner, survey_tr, y[tr], band_tr, centre)
        band_va = ((out[STAGE1_ONLY][va] > BAND_LO)
                   & (out[STAGE1_ONLY][va] < BAND_HI))
        score = stage1_va.copy()
        score[band_va] += lam_c * (survey_va[band_va] - centre)
        out[CASCADE][va] = expit(score)

        # --- full + 帯の中だけアンケートを二重計上 ---------------------------
        band_full_tr = (expit(full_inner) > BAND_LO) & (expit(full_inner) < BAND_HI)
        lam_p = _choose_lambda(full_inner, survey_tr, y[tr], band_full_tr, centre)
        band_full_va = (out[FULL][va] > BAND_LO) & (out[FULL][va] < BAND_HI)
        score = full_va.copy()
        score[band_full_va] += lam_p * (survey_va[band_full_va] - centre)
        out[FULL_PLUS][va] = expit(score)

        # --- アンケートの重みを帯の内外で分けてメタに推定させる ------------
        # 「帯の中ではアンケートを重く」が正しいなら、交互作用列に正の重みが
        # 付くはず。非負制約があるので符号を選べるよう正負両方渡す。
        def _augment(Z_block, band_mask):
            inter = Z_block[:, survey_col] * band_mask.astype(float)
            inter = (inter - inter.mean()) / (inter.std() + 1e-9)
            return np.column_stack([Z_block, inter, -inter])

        aug_tr = _augment(Z_all[tr], band_full_tr)
        aug_va = _augment(Z_all[va], band_full_va)
        alpha_s, w_s, b_s = _fit_meta_on(aug_tr, y[tr], seed)
        out[SPLIT][va] = expit(aug_va @ w_s + b_s)

        detail.append({
            "seed": seed,
            "交互作用の重み(+側)": float(w_s[-2]),
            "交互作用の重み(-側)": float(w_s[-1]),
            "E2本体の重み": float(w_s[survey_col]),
            "1段目の帯 n": int(band_tr.sum()),
            "現行の帯 n": int(band_full_tr.sum()),
            "lambda[cascade]": lam_c,
            "lambda[full_plus]": lam_p,
            "1段目の帯内実購入率": float(y[tr][band_tr].mean()) if band_tr.sum() else np.nan,
            "帯内アンケート単体AUC": (
                roc_auc_score(y[tr][band_tr], survey_tr[band_tr])
                if band_tr.sum() and len(np.unique(y[tr][band_tr])) > 1 else np.nan),
        })
    return out, detail


def main(seeds, config):
    rows, details = [], []
    for seed in seeds:
        y, Z, _ = _load(CONFIGS[config], seed)
        out, detail = _one_seed(y, Z, seed)
        details.extend(detail)
        for arm, p in out.items():
            rows.append({"seed": seed, "arm": arm,
                         "auc": roc_auc_score(y, p),
                         "ap": average_precision_score(y, p),
                         "f1": _best_f1(y, p)})
    result = pd.DataFrame(rows)
    frame = pd.DataFrame(details)

    print(f"1段目の構成: {STAGE1}")
    print(f"2段目に回す: E2_survey（E7_cross は入力がアンケート由来なので1段目から除外）")
    print("\n=== fold ごとの内部量（平均）===")
    print(frame.mean(numeric_only=True).to_string(
        float_format=lambda v: f"{v:.4f}"))
    for key in ("lambda[cascade]", "lambda[full_plus]"):
        counts = frame[key].value_counts().sort_index()
        print(f"  {key:22s} " +
              "  ".join(f"λ={k:g}: {v}fold" for k, v in counts.items()))

    print("\n=== 本番と同じ外側5fold での成績 ===")
    print(result.groupby("arm")[["auc", "ap", "f1"]].mean().reindex(
        list(ARMS)).to_string(float_format=lambda v: f"{v:.5f}"))
    base = result[result.arm == FULL].sort_values("seed").reset_index(drop=True)
    print(f"\n=== {FULL} を基準にしたペア差 ===")
    for arm in (STAGE1_ONLY, CASCADE, FULL_PLUS, SPLIT):
        cand = result[result.arm == arm].sort_values("seed").reset_index(drop=True)
        line = [f"  {arm:22s}"]
        for metric in ("auc", "ap", "f1"):
            d = cand[metric].values - base[metric].values
            line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
        decision, _ = verdict(base, cand)
        print("  ".join(line) + f"   {decision}")

    ref = result[result.arm == STAGE1_ONLY].sort_values("seed").reset_index(drop=True)
    cascade = result[result.arm == CASCADE].sort_values("seed").reset_index(drop=True)
    print(f"\n=== {CASCADE} - {STAGE1_ONLY}（2段目が1段目に何を足したか）===")
    line = ["  "]
    for metric in ("auc", "ap", "f1"):
        d = cascade[metric].values - ref[metric].values
        line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
    print("  ".join(line))

    result.to_csv("exp/survey_cascade.csv", index=False)
    print("\nsaved: exp/survey_cascade.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 42])
    args = parser.parse_args()
    main(args.seeds, args.config)
