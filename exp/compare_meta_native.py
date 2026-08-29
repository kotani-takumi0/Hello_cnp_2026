"""H70-H76: candidates that only exist once meta contribution is the criterion.

Every hypothesis so far built an expert by fitting it to ``y`` on its own and
then measured what the meta gained.  The audit of 2026-08-29 changed the
verdict rule to meta contribution but left the *search space* unchanged, so the
same saturated family kept being resampled.  This script screens the moves that
the old standalone-AP gate could not express:

  base          current non-negative logistic meta over eight expert logits
  H70 bag_meta      bagged meta weights; same class, less weight variance
  H71 disagree      expert disagreement as a meta input and as a temperature
  H72 band_weight   meta refit with the ambiguous band upweighted
  H73 band_gate     non-negative additive correction, inside the band only
  H74 resid_boost   LightGBM trained on the blend logit as offset
  H75 resid_expert  that residual score entered as a ninth meta column
  H76 slot_swap     E7 (no unique information) swapped for the residual column

The first 5-repeat screen sent the H73 gain back to its cause: it appeared only
in the two repeats where ``select_alpha`` picked alpha=1.0, over-shrank the
weights, and left the blend badly miscalibrated.  That is a defect in alpha
selection, not a property of the band, so four more arms test the selection
itself:

  H77 alpha_1se      smallest alpha within one standard error of the best
  H78 alpha_fix003   no selection; a fixed alpha in the flat part of the curve
  H79 alpha_avg      weights averaged over the flat part instead of selected
  H80 platt          base plus a two-parameter Platt fit on A (temperature)

The 20-repeat run then split by whether alpha had blown up in that repeat, and
H74 was +0.014 AP with 9/9 consistency in the 9 healthy repeats and negative in
the 11 broken ones.  That split conditions on a property of the baseline, so it
cannot decide anything by itself.  Two arms redo H73 and H74 on a base whose
alpha is already fixed, which turns the post-hoc split into a real 2x2:

  H81 1se_band_gate    H73 on top of the alpha_1se meta
  H82 1se_resid_boost  H74 on top of the alpha_1se meta
  H83 1se_resid_placebo  H82 with the residual model's feature rows shuffled

H83 is the placebo for H82.  Shuffling the feature rows against the labels
destroys any x -> residual mapping while leaving the offset, the labels, the
tree budget, and the additive form untouched, so whatever H83 gains is what the
construction gives for free rather than what the features carry.

H80 is the control that separates the two damages: Platt is monotone, so it can
repair the temperature but never the ranking.  Anything the alpha arms gain in
AUC/AP above H80 is ranking that alpha selection was throwing away.

Outcome.  Only the alpha family survived.  H82 reached +0.0199 AP with 20/20
sign consistency here and still turned out to be a false positive: on the
production fold structure it is -0.0063 AP (1/11), worse than its own placebo.
The tell was already in this script's output -- the A-side cross-fit column
``train_ap`` moves the opposite way (-0.0022, 6/20) from the B-side ``ap``.
Fitting and applying use experts trained on different amounts of data (about
385 rows for the A-side OOF against 482 for the refit used on B), so a stage
that learns the blend's own error structure is rewarded for a shift that does
not exist in production.  Check ``train_ap`` against ``ap`` before believing any
arm that takes the blend's output as an input.  See
documents/meta_native_and_alpha_selection_20260829.md section 4.4.

All arms share one expert-fitting pass per repeat, so every comparison is
exactly paired: the eight expert predictions are bit-identical across arms and
only the combiner differs.

Leakage control.  B labels are used exactly once, to score.  Alpha, weights,
the band definition, the residual model, and the threshold all come from A.
The A-side score that each threshold rule sees is itself a meta-level 5-fold
cross-fit, so an arm with more capacity cannot buy an optimistic threshold --
without this the added-capacity arms would be scored at a threshold fitted to
their own in-sample overshoot.

Examples:
  .venv/bin/python exp/compare_meta_native.py --n-reps 1 --seed-offset 6999
  .venv/bin/python exp/compare_meta_native.py --n-reps 5 --seed-offset 700
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from decision import verdict  # noqa: E402
from harness import LGB_THREADS  # noqa: E402
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import (  # noqa: E402
    ALPHA_GRID, EPS, apply_meta, fit_meta, select_alpha, select_alpha_1se,
)
from meta_pairwise import fit_platt  # noqa: E402
from metric_suite import score_predictions  # noqa: E402

BASELINE = "base"
ARMS = (BASELINE, "bag_meta", "disagree", "band_weight", "band_gate",
        "resid_boost", "resid_expert", "slot_swap",
        "alpha_1se", "alpha_fix003", "alpha_avg", "platt",
        "1se_band_gate", "1se_resid_boost", "1se_resid_placebo")

# H78/H79: 内側AP曲線が平らな区間（diagnose_meta_alpha.py の実測で
# 0.87523 / 0.87612 / 0.87641 / 0.87496 と 0.0015 しか動かない）。
FLAT_GRID = (0.001, 0.003, 0.01, 0.03)
FIXED_ALPHA = 0.003

# 曖昧帯の定義。uncertain_band_profile_20260828.md と同じ切り方。
BAND_LO, BAND_HI = 0.1, 0.9
# H72: 帯の行に与える追加重み（1 + この値）。
BAND_EXTRA_WEIGHT = 3.0
# H70: メタ重みのブートストラップ本数。8次元の凸フィットなので安く回る。
N_BAG = 100
# H74-H76: 残差モデル。742行の上に載せるので容量を強く絞る。
RESID_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=4,
    min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.7,
    bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=LGB_THREADS,
)
RESID_ROUNDS = 150
THRESHOLD_RULES = ("oof_best", "half_f1")


# --------------------------------------------------------------------------
# 重み付き・オフセット付きのメタ学習器
#
# meta_blend.fit_meta と同じ損失で、行重みと行ごとのオフセットを足せるように
# しただけ。重み一様・オフセット0なら fit_meta と数値的に一致する（下の
# --self-check で検証できる）。
# --------------------------------------------------------------------------
def _weighted_loss_grad(theta, Z, y, alpha, weight, offset):
    w, b = theta[:-1], theta[-1]
    p = np.clip(expit(Z @ w + b + offset), EPS, 1.0 - EPS)
    loss = -np.sum(weight * (y * np.log(p) + (1 - y) * np.log1p(-p)))
    r = weight * (p - y)
    return (loss + alpha * (w @ w),
            np.concatenate([Z.T @ r + 2.0 * alpha * w, [r.sum()]]))


def fit_meta_weighted(Z, y, alpha=1.0, weight=None, offset=None):
    """非負重み + 切片。weight は内部で総和1に正規化する。"""
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    weight = weight / weight.sum()
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    k = Z.shape[1]
    theta0 = np.concatenate([np.full(k, 1.0 / k), [0.0]])
    res = minimize(_weighted_loss_grad, theta0,
                   args=(Z, y, alpha, weight, offset), jac=True,
                   method="L-BFGS-B",
                   bounds=[(0.0, None)] * k + [(None, None)])
    return res.x[:-1], res.x[-1]


def select_alpha_weighted(Z, y, seed, weight=None, offset=None,
                          eval_mask=None, grid=ALPHA_GRID, n_splits=5):
    """渡された行の中だけで alpha を選ぶ。基準は AP（select_alpha と同じ）。

    eval_mask を渡すと、その部分集合の AP で選ぶ。H73 の補正項は帯の中でしか
    使われないので、帯の外の行で alpha を決めても意味がない。
    """
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    folds = list(StratifiedKFold(n_splits, shuffle=True,
                                 random_state=seed).split(Z, y))
    mask = np.ones(len(y), bool) if eval_mask is None else eval_mask
    best, best_ap = grid[0], -np.inf
    for a in grid:
        oof = np.zeros(len(y))
        for tr, va in folds:
            w, b = fit_meta_weighted(
                Z[tr], y[tr], alpha=a,
                weight=None if weight is None else weight[tr],
                offset=offset[tr])
            oof[va] = expit(Z[va] @ w + b + offset[va])
        if len(np.unique(y[mask])) < 2:
            continue
        ap = average_precision_score(y[mask], oof[mask])
        if ap > best_ap:
            best, best_ap = a, ap
    return best


def average_path_meta(Z, y, grid=FLAT_GRID):
    """平らな区間の alpha をまたいで重みを平均する。選択そのものを止める版。"""
    weights, intercepts = [], []
    for alpha in grid:
        w, b = fit_meta(Z, y, alpha=alpha)
        weights.append(w)
        intercepts.append(b)
    return np.mean(weights, axis=0), float(np.mean(intercepts))


def _oof_logit(Z, y, alpha, seed, n_splits=5):
    """与えられた行集合の中だけで、基準メタの OOF ロジットを作る。

    残差モデルのオフセットと帯の定義に使う。in-sample のロジットを使うと
    残差が人工的に縮んで、残差モデルが「もう直すものは無い」と学習してしまう。
    """
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True,
                                  random_state=seed).split(Z, y):
        w, b = fit_meta(Z[tr], y[tr], alpha=alpha)
        out[va] = Z[va] @ w + b
    return out


def _standardize(columns, reference):
    """reference（学習側）の平均・標準偏差で列を標準化する。"""
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0) + 1e-9
    return (columns - mu) / sd


def _disagreement_columns(Z):
    """行ごとの不一致量と、それが合成スコアを鈍らせる向きの交互作用。

    非負制約があるので符号を選べない。同じ列の正負両方を渡して、必要な向きを
    メタ自身に選ばせる（不要な側は重み0に落ちる）。
    """
    spread = Z.std(axis=1)
    return np.column_stack([spread, spread * Z.mean(axis=1)])


def _residual_model(X_fit, y_fit, offset, seed):
    return lgb.train(
        {**RESID_PARAMS, "seed": seed},
        lgb.Dataset(X_fit, y_fit, init_score=offset),
        num_boost_round=RESID_ROUNDS,
    )


def _band_and_residual(Z_fit, y_fit, Z_app, X_fit, X_app, alpha, weight,
                       intercept, seed):
    """あるメタ（重み・切片）を土台に、帯ゲートと残差ブーストを載せる。

    土台を差し替えて同じ手続きを繰り返せるように切り出してある。H73/H74 は
    現行 alpha の土台の上で、H81/H82 は alpha_1se の土台の上で呼ばれる。
    """
    app_logit = Z_app @ weight + intercept
    oof_logit = _oof_logit(Z_fit, y_fit, alpha, seed)
    band_fit = ((expit(oof_logit) > BAND_LO) & (expit(oof_logit) < BAND_HI))
    app_probability = expit(app_logit)
    band_app = (app_probability > BAND_LO) & (app_probability < BAND_HI)

    if band_fit.sum() >= 40 and len(np.unique(y_fit[band_fit])) == 2:
        alpha_g = select_alpha_weighted(
            Z_fit, y_fit, seed, weight=band_fit.astype(float) + 1e-9,
            offset=oof_logit, eval_mask=band_fit)
        w_g, b_g = fit_meta_weighted(
            Z_fit[band_fit], y_fit[band_fit], alpha=alpha_g,
            offset=oof_logit[band_fit])
        gate = expit(app_logit + band_app * (Z_app @ w_g + b_g))
        gate_norm = float(np.abs(w_g).sum())
    else:
        gate, gate_norm = expit(app_logit), 0.0

    model = _residual_model(X_fit, y_fit, oof_logit, seed)
    delta_app = model.predict(X_app, raw_score=True)
    return {
        "band_gate": gate, "resid_boost": expit(app_logit + delta_app),
        "band_rate": float(band_fit.mean()), "gate_norm": gate_norm,
        "resid_delta_std": float(delta_app.std()),
        "oof_logit": oof_logit, "delta_app": delta_app,
    }


def _arms(X, y, fit_idx, Z_fit, app_idx, Z_app, resid_cols, seed):
    """fit_idx で全アームを学習し、app_idx の行の確率を返す。

    app_idx のラベルは一切参照しない。この関数は「A内のメタCV」と
    「A全体で学習してBを予測」の両方から同じ形で呼ばれる。
    """
    y_fit = y[fit_idx]
    out, detail = {}, {}

    # --- base: 現行メタ ---------------------------------------------------
    alpha = select_alpha(Z_fit, y_fit, seed=seed)[0]
    weight, intercept = fit_meta(Z_fit, y_fit, alpha=alpha)
    base_app_logit = Z_app @ weight + intercept
    out[BASELINE] = expit(base_app_logit)
    detail["alpha"] = alpha

    # --- H70 bag_meta: 重み推定の分散を落とす -----------------------------
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_fit == 1)
    negative = np.flatnonzero(y_fit == 0)
    bagged = np.zeros(Z_fit.shape[1])
    bagged_intercept = 0.0
    for _ in range(N_BAG):
        # 層化ブートストラップ。正例が少ないので単純復元抽出だと正例ゼロの
        # 標本が出て fit が壊れる。
        pick = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        w_b, b_b = fit_meta(Z_fit[pick], y_fit[pick], alpha=alpha)
        bagged += w_b
        bagged_intercept += b_b
    bagged /= N_BAG
    bagged_intercept /= N_BAG
    out["bag_meta"] = expit(Z_app @ bagged + bagged_intercept)
    detail["bag_weight_shift"] = float(np.abs(bagged - weight).sum())

    # --- H71 disagree: 不一致量をメタ入力に足す ---------------------------
    raw_fit = _disagreement_columns(Z_fit)
    raw_app = _disagreement_columns(Z_app)
    std_fit = _standardize(raw_fit, raw_fit)
    std_app = _standardize(raw_app, raw_fit)
    aug_fit = np.column_stack([Z_fit, std_fit, -std_fit])
    aug_app = np.column_stack([Z_app, std_app, -std_app])
    alpha_d = select_alpha(aug_fit, y_fit, seed=seed)[0]
    w_d, b_d = fit_meta(aug_fit, y_fit, alpha=alpha_d)
    out["disagree"] = apply_meta(aug_app, w_d, b_d)
    detail["disagree_weight"] = float(w_d[Z_fit.shape[1]:].sum())

    # --- H73 band_gate / H74 resid_boost: 現行alphaの土台の上で -----------
    X_fit = X.iloc[fit_idx][resid_cols]
    X_app = X.iloc[app_idx][resid_cols]
    on_base = _band_and_residual(Z_fit, y_fit, Z_app, X_fit, X_app, alpha,
                                 weight, intercept, seed)
    out["band_gate"] = on_base["band_gate"]
    out["resid_boost"] = on_base["resid_boost"]
    oof_logit, delta_app = on_base["oof_logit"], on_base["delta_app"]
    band_fit = (expit(oof_logit) > BAND_LO) & (expit(oof_logit) < BAND_HI)
    detail["band_rate_fit"] = on_base["band_rate"]
    detail["gate_norm"] = on_base["gate_norm"]
    detail["resid_delta_std"] = on_base["resid_delta_std"]

    # --- H72 band_weight: 帯の行を重く見て同じ形のメタを引き直す ----------
    row_weight = 1.0 + BAND_EXTRA_WEIGHT * band_fit
    alpha_w = select_alpha_weighted(Z_fit, y_fit, seed, weight=row_weight)
    w_w, b_w = fit_meta_weighted(Z_fit, y_fit, alpha=alpha_w,
                                 weight=row_weight)
    out["band_weight"] = expit(Z_app @ w_w + b_w)

    # --- H75/H76 残差スコアを9本目の列として渡す --------------------------
    delta_fit = np.zeros(len(y_fit))
    for tr, va in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(Z_fit, y_fit):
        fold_model = _residual_model(
            X_fit.iloc[tr], y_fit[tr], oof_logit[tr], seed)
        delta_fit[va] = fold_model.predict(X_fit.iloc[va], raw_score=True)
    stacked_fit = np.column_stack([Z_fit, delta_fit])
    stacked_app = np.column_stack([Z_app, delta_app])
    alpha_r = select_alpha(stacked_fit, y_fit, seed=seed)[0]
    w_r, b_r = fit_meta(stacked_fit, y_fit, alpha=alpha_r)
    out["resid_expert"] = apply_meta(stacked_app, w_r, b_r)
    detail["resid_expert_weight"] = float(w_r[-1])

    keep = [i for i, name in enumerate(EXPERTS) if name != "E7_cross"]
    keep.append(len(EXPERTS))
    alpha_s = select_alpha(stacked_fit[:, keep], y_fit, seed=seed)[0]
    w_s, b_s = fit_meta(stacked_fit[:, keep], y_fit, alpha=alpha_s)
    out["slot_swap"] = apply_meta(stacked_app[:, keep], w_s, b_s)

    # --- H77 alpha_1se: 平らな区間では小さい側に留まる ---------------------
    alpha_1se = select_alpha_1se(Z_fit, y_fit, seed)
    w_1, b_1 = fit_meta(Z_fit, y_fit, alpha=alpha_1se)
    out["alpha_1se"] = expit(Z_app @ w_1 + b_1)
    detail["alpha_1se"] = alpha_1se

    # --- H78 alpha_fix003: そもそも選ばない ------------------------------
    w_f, b_f = fit_meta(Z_fit, y_fit, alpha=FIXED_ALPHA)
    out["alpha_fix003"] = expit(Z_app @ w_f + b_f)

    # --- H79 alpha_avg: 平らな区間をまたいで平均する ---------------------
    w_a, b_a = average_path_meta(Z_fit, y_fit)
    out["alpha_avg"] = expit(Z_app @ w_a + b_a)

    # --- H80 platt: 温度だけ直す対照 --------------------------------------
    scale, shift = fit_platt(oof_logit, y_fit)
    out["platt"] = expit(scale * base_app_logit + shift)
    detail["platt_scale"] = scale

    # --- H81/H82: alpha を直したベースの上でH73/H74をやり直す --------------
    # 20repの内訳で、H74は「alphaが暴走しなかった9repだけ 9/9 で +0.014 AP」と
    # 出た。それは base の性質で切った事後分割なので判定には使えない。
    # ベース側を alpha_1se に固定した2x2で測り直す。
    on_1se = _band_and_residual(
        Z_fit, y_fit, Z_app, X_fit, X_app, alpha_1se, w_1, b_1, seed)
    out["1se_band_gate"] = on_1se["band_gate"]
    out["1se_resid_boost"] = on_1se["resid_boost"]
    detail["1se_band_rate"] = on_1se["band_rate"]
    detail["1se_resid_delta_std"] = on_1se["resid_delta_std"]

    # --- H83: H82のプラセボ。特徴の行だけシャッフルして写像を壊す ---------
    shuffled = X_fit.iloc[
        np.random.default_rng(seed + 991).permutation(len(X_fit))
    ].reset_index(drop=True)
    placebo = _residual_model(shuffled, y_fit, on_1se["oof_logit"], seed)
    delta_placebo = placebo.predict(X_app, raw_score=True)
    out["1se_resid_placebo"] = expit(Z_app @ w_1 + b_1 + delta_placebo)
    detail["placebo_delta_std"] = float(delta_placebo.std())

    e2 = EXPERTS.index("E2_survey")
    for name, vector in (("base", weight), ("fix003", w_f), ("1se", w_1)):
        total = vector[:len(EXPERTS)].sum()
        detail[f"E2_share_{name}"] = (
            float(vector[e2] / total) if total > 0 else 0.0)
    return out, detail


def _one_repeat(X, y, text, org, e4, columns, seed, test_size):
    all_indices = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_holdout = y[train_idx], y[holdout_idx]

    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed).split(train_idx, y_train))
    expert_oof = {name: np.zeros(len(train_idx)) for name in EXPERTS}
    for inner_train, inner_valid in folds:
        predictions = fit_predict_experts(
            X, y, text, org, train_idx[inner_train], train_idx[inner_valid],
            seed, columns, e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C,
            strict_lgbm=True)
        for name in EXPERTS:
            expert_oof[name][inner_valid] = predictions[name]
    expert_holdout = fit_predict_experts(
        X, y, text, org, train_idx, holdout_idx, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True)

    Z_train = _stack(expert_oof, EXPERTS)
    Z_holdout = _stack(expert_holdout, EXPERTS)
    resid_cols = columns[0]

    # A 側スコアはメタ層の交差適合で作る。容量の大きいアームが自分の
    # in-sample の外れ値に合わせた閾値をもらわないようにするため。
    train_probability = {arm: np.zeros(len(train_idx)) for arm in ARMS}
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed + 777).split(Z_train, y_train):
        inner, _ = _arms(X, y, train_idx[tr], Z_train[tr], train_idx[va],
                         Z_train[va], resid_cols, seed)
        for arm, probability in inner.items():
            train_probability[arm][va] = probability

    holdout_probability, detail = _arms(
        X, y, train_idx, Z_train, holdout_idx, Z_holdout, resid_cols, seed)

    rows = []
    for arm in ARMS:
        thresholds = _thresholds(train_probability[arm], y_train,
                                 holdout_probability[arm], folds,
                                 THRESHOLD_RULES)
        row = score_predictions(y_holdout, holdout_probability[arm],
                                thresholds["oof_best"])
        row["f1_half_f1"] = f1_score(
            y_holdout,
            (holdout_probability[arm] >= thresholds["half_f1"]).astype(int))
        row["train_ap"] = average_precision_score(
            y_train, train_probability[arm])
        rows.append({"seed": seed, "arm": arm, **row})
    return rows, detail


METRICS = ("auc", "ap", "f1", "f1_half_f1", "precision", "recall",
           "logloss", "brier", "pos_rate", "threshold", "train_ap")


def _report(frame):
    print("\n=== strict holdout: mean over repeats ===")
    print(frame.groupby("arm")[list(METRICS)].mean().T.to_string(
        float_format=lambda v: f"{v:.5f}"))
    pivot = frame.pivot(index="seed", columns="arm")
    base_frame = frame[frame["arm"] == BASELINE].sort_values("seed")
    for arm in ARMS:
        if arm == BASELINE:
            continue
        print(f"\n=== {arm} - {BASELINE} ===")
        for metric in ("auc", "ap", "f1", "f1_half_f1"):
            delta = pivot[(metric, arm)] - pivot[(metric, BASELINE)]
            print(f"  {metric:12s} mean {delta.mean():+.5f}  "
                  f"positive {int((delta > 0).sum())}/{len(delta)}  "
                  f"std {delta.std():.5f}")
        candidate = frame[frame["arm"] == arm].sort_values("seed")
        decision, _ = verdict(base_frame.reset_index(drop=True),
                              candidate.reset_index(drop=True))
        print(f"  判定: {decision}")


def run(n_reps, seed_offset, test_size, output):
    X, y, text, org, e4, columns = _inputs()
    records, details = [], []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, detail = _one_repeat(
            X, y, text, org, e4, columns, seed, test_size)
        records.extend(rows)
        details.append({"seed": seed, **detail})
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")
    print("\n=== arm diagnostics (A側で学習したときの内部量) ===")
    print(pd.DataFrame(details).to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}"))
    _report(frame)
    return frame


def _self_check():
    """fit_meta_weighted が一様重みで fit_meta と一致することを確認する。"""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 8))
    y = (rng.random(200) < expit(Z[:, 0])).astype(int)
    for alpha in (0.001, 0.1):
        w1, b1 = fit_meta(Z, y, alpha=alpha)
        w2, b2 = fit_meta_weighted(Z, y, alpha=alpha)
        assert np.allclose(w1, w2, atol=1e-6), (alpha, w1, w2)
        assert np.isclose(b1, b2, atol=1e-6), (alpha, b1, b2)
    print("self-check OK: fit_meta_weighted == fit_meta (一様重み)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=700)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="重み付きメタ学習器の一致確認だけして終了")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
    else:
        out = args.out_csv or (
            f"exp/holdout_meta_native_seed{args.seed_offset}"
            f"_{args.n_reps}rep.csv")
        run(args.n_reps, args.seed_offset, args.test_size, out)
