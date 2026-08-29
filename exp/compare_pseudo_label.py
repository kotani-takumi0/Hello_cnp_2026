"""H84: attack the one axis never attacked -- the number of training rows.

Every hypothesis so far changed what the experts read or how they are combined.
None changed how many rows they learn from.  ``bayes_ceiling_and_new_angles``
listed this as candidate 2 ("n軸は未攻略") and it was never run.

The 800 test rows are unlabelled, but the current blend is extremely confident
about most of them: on train, rows with p<0.1 carry a 1.1% error rate and rows
with p>0.9 carry 3.8%.  Pseudo-labelling those and adding them to the training
set takes the experts from 742 rows to roughly 1300 -- a 76% increase.  The
holdout learning curve makes that worth testing: 482 rows give F1 0.798 and 742
rows give 0.82-0.83, so the slope over this range is not flat.

Protocol per repeat -- B labels are never used, and the pseudo-labels come from
a model that has only seen A:

  1. split the 742 labelled rows into A and B
  2. fit every expert on A, fit the meta on A's OOF, predict all 800 test rows
  3. keep the test rows outside [lo, hi] and give them hard labels
  4. rebuild A's OOF and the B predictions with A + pseudo as the training set
  5. score on B

Test rows have no labels at all, so step 3 cannot leak; step 4 adds rows to
expert training only, leaving the meta fit on A's real labels.

The pseudo-label error rate is estimated honestly by applying the same
confidence rule to B, whose labels are known but never used for fitting.  It is
reported as a diagnostic and takes no part in any decision.

Examples:
  .venv/bin/python exp/compare_pseudo_label.py --n-reps 1 --seed-offset 8999
  .venv/bin/python exp/compare_pseudo_label.py --n-reps 5 --seed-offset 900
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS  # noqa: E402
from decision import verdict  # noqa: E402
from ensemble_experts import build_features, e4_inputs  # noqa: E402
from expert_groups import (  # noqa: E402
    FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols, e1_finance_cols,
    e2_survey_cols, e6_manual_cols, e7_cross_cols, route_h2_anchor_cols,
)
from holdout_check import _stack, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha_1se  # noqa: E402
from metric_suite import score_predictions  # noqa: E402
from organization_features import load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402

BASELINE = "train_only"
CANDIDATE = "plus_pseudo"
BALANCED = "plus_pseudo_balanced"
ARMS = (BASELINE, CANDIDATE, BALANCED)
THRESHOLD_RULES = ("oof_best", "half_f1")


def _inputs_with_test():
    """train と test を縦に連結した特徴を返す。

    ``fit_predict_experts`` は X/y/txt/org/e4 を tr・target で添字するだけなので、
    連結した配列を渡せば test 行を学習側に混ぜられる。前半742行が train、
    後半800行が test。
    """
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, X_test = build_features(
        train, test, cross=True, llm_makebuy=True,
        h2_restore=True, h7_restore=True)
    e4_train, e4_test, e4_model = e4_inputs(
        train, test, org_embed=False, concat_embed=True)

    X_all = pd.concat([X, X_test], ignore_index=True)
    for column in X.columns:
        assert X_all[column].dtype == X[column].dtype, (
            f"連結で型が変わった: {column} "
            f"{X[column].dtype} -> {X_all[column].dtype}")
    text_all = np.concatenate([
        train[TEXT_COL].fillna("").astype(str).values,
        test[TEXT_COL].fillna("").astype(str).values])
    org_all = np.concatenate([load_org_text(train), load_org_text(test)])
    e4_all = np.vstack([e4_train, e4_test])
    assert len(X_all) == len(text_all) == len(org_all) == len(e4_all)

    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    core4 = {
        "LLM_need_score", "LLM_make_limitation_score",
        "LLM_buy_openness_score", "LLM_internal_training_capability_score",
    }
    c0 = [
        c for c in e0_anchor_cols(X.columns)
        if not c.startswith("DX展望_") and c not in missing
        and (not c.startswith("LLM_") or c in core4)
    ]
    c0b = [c for c in e0_anchor_cols(X.columns) if not c.startswith("LLM_")]
    c0, c0b = route_h2_anchor_cols(c0, c0b, "h2h7_both4")
    columns = (
        c0, c0b, e1_finance_cols(), e2_survey_cols(),
        e6_manual_cols(X.columns), e7_cross_cols(),
    )
    return X_all, y, text_all, org_all, (e4_all, e4_model), columns, len(X)


def _fit_all(X, y, text, org, e4, columns, train_rows, target_rows, seed):
    return fit_predict_experts(
        X, y, text, org, train_rows, target_rows, seed, columns,
        e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True)


def _meta(Z_fit, y_fit, Z_apply, seed):
    alpha = select_alpha_1se(Z_fit, y_fit, seed)
    weight, intercept = fit_meta(Z_fit, y_fit, alpha=alpha)
    return (apply_meta(Z_fit, weight, intercept),
            apply_meta(Z_apply, weight, intercept))


def _one_repeat(X, y_train, text, org, e4, columns, n_train, seed,
                test_size, low, high):
    labelled = np.arange(n_train)
    test_rows = np.arange(n_train, len(X))
    A, B = train_test_split(
        labelled, test_size=test_size, stratify=y_train, random_state=seed)
    y_A, y_B = y_train[A], y_train[B]

    # y は連結配列と同じ長さで持つ。test 側は疑似ラベルで埋める。
    y_all = np.zeros(len(X), dtype=int)
    y_all[labelled] = y_train

    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed).split(A, y_A))

    # --- 基準: A だけで学習 -------------------------------------------------
    oof = {name: np.zeros(len(A)) for name in EXPERTS}
    for fold_train, fold_valid in folds:
        p = _fit_all(X, y_all, text, org, e4, columns,
                     A[fold_train], A[fold_valid], seed)
        for name in EXPERTS:
            oof[name][fold_valid] = p[name]
    Z_A = _stack(oof, EXPERTS)
    base_B = _fit_all(X, y_all, text, org, e4, columns, A, B, seed)
    base_train_p, base_B_p = _meta(Z_A, y_A, _stack(base_B, EXPERTS), seed)

    # --- 疑似ラベルを作る（A だけを見たモデルで test 800行を予測）----------
    test_pred = _fit_all(X, y_all, text, org, e4, columns, A, test_rows, seed)
    _, p_test = _meta(Z_A, y_A, _stack(test_pred, EXPERTS), seed)
    confident = (p_test < low) | (p_test > high)
    pseudo_rows = test_rows[confident]
    y_all[pseudo_rows] = (p_test[confident] > 0.5).astype(int)

    # 同じ信頼規則を B に当てて疑似ラベルの誤り率を推定する（診断のみ）。
    band_B = (base_B_p < low) | (base_B_p > high)
    pseudo_error = (
        float(((base_B_p[band_B] > 0.5).astype(int) != y_B[band_B]).mean())
        if band_B.sum() else np.nan)

    # 信頼できる行は圧倒的に負例側に偏る（確信-非購入の方が広いため）ので、
    # そのまま足すと学習集合の正例率が下がる。負例を間引いて train と同じ
    # 正例率に揃えた版も並べて、偏りが原因かどうかを切り分ける。
    pseudo_y = y_all[pseudo_rows]
    positives = pseudo_rows[pseudo_y == 1]
    negatives = pseudo_rows[pseudo_y == 0]
    target_rate = float(y_A.mean())
    keep_negative = int(min(len(negatives),
                            round(len(positives) * (1 - target_rate) / target_rate)))
    balanced_rows = np.concatenate([
        positives,
        np.random.default_rng(seed).choice(negatives, keep_negative, replace=False),
    ]) if len(positives) and keep_negative else pseudo_rows

    # --- 候補: A + 疑似ラベル で学習 ---------------------------------------
    fitted = {}
    for name, extra in ((CANDIDATE, pseudo_rows), (BALANCED, balanced_rows)):
        oof_p = {n: np.zeros(len(A)) for n in EXPERTS}
        for fold_train, fold_valid in folds:
            p = _fit_all(X, y_all, text, org, e4, columns,
                         np.concatenate([A[fold_train], extra]),
                         A[fold_valid], seed)
            for n in EXPERTS:
                oof_p[n][fold_valid] = p[n]
        pred_B = _fit_all(X, y_all, text, org, e4, columns,
                          np.concatenate([A, extra]), B, seed)
        fitted[name] = _meta(_stack(oof_p, EXPERTS), y_A,
                             _stack(pred_B, EXPERTS), seed)

    rows = []
    for name, (train_p, hold_p) in ((BASELINE, (base_train_p, base_B_p)),
                                    (CANDIDATE, fitted[CANDIDATE]),
                                    (BALANCED, fitted[BALANCED])):
        th = _thresholds(train_p, y_A, hold_p, folds, THRESHOLD_RULES)
        row = score_predictions(y_B, hold_p, th["oof_best"])
        row["f1_half_f1"] = score_predictions(
            y_B, hold_p, th["half_f1"])["f1"]
        rows.append({"seed": seed, "arm": name, **row})
    detail = {
        "seed": seed, "n_pseudo": int(len(pseudo_rows)),
        "n_pseudo_balanced": int(len(balanced_rows)),
        "pseudo_pos_rate": float(y_all[pseudo_rows].mean()) if len(pseudo_rows) else np.nan,
        "train_rows_before": len(A),
        "train_rows_after": len(A) + len(pseudo_rows),
        "pseudo_error_on_B": pseudo_error,
    }
    return rows, detail


METRICS = ("auc", "ap", "f1", "f1_half_f1", "precision", "recall", "logloss")


def run(n_reps, seed_offset, test_size, low, high, output):
    X, y, text, org, e4, columns, n_train = _inputs_with_test()
    print(f"連結後 {len(X)}行（train {n_train} + test {len(X) - n_train}）")
    records, details = [], []
    for repeat in range(n_reps):
        seed = seed_offset + repeat
        print(f"repeat {repeat + 1}/{n_reps} (seed={seed})", flush=True)
        rows, detail = _one_repeat(
            X, y, text, org, e4, columns, n_train, seed, test_size, low, high)
        records.extend(rows)
        details.append(detail)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    print(f"saved: {output}")

    print("\n=== 疑似ラベルの内訳 ===")
    print(pd.DataFrame(details).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== strict holdout: mean over repeats ===")
    print(frame.groupby("arm")[list(METRICS)].mean().T.to_string(
        float_format=lambda v: f"{v:.5f}"))
    pivot = frame.pivot(index="seed", columns="arm")
    base = frame[frame["arm"] == BASELINE].sort_values("seed")
    for arm in (CANDIDATE, BALANCED):
        print(f"\n=== {arm} - {BASELINE} ===")
        for metric in ("auc", "ap", "f1", "f1_half_f1"):
            delta = pivot[(metric, arm)] - pivot[(metric, BASELINE)]
            print(f"  {metric:12s} mean {delta.mean():+.5f}  "
                  f"positive {int((delta > 0).sum())}/{len(delta)}  "
                  f"std {delta.std():.5f}")
        cand = frame[frame["arm"] == arm].sort_values("seed")
        decision, _ = verdict(base.reset_index(drop=True),
                              cand.reset_index(drop=True))
        print(f"  判定: {decision}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=900)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--low", type=float, default=0.1,
                        help="この確率未満の test 行を負例として採用")
    parser.add_argument("--high", type=float, default=0.9,
                        help="この確率超の test 行を正例として採用")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()
    out = args.out_csv or (
        f"exp/holdout_pseudo_label_seed{args.seed_offset}"
        f"_{args.n_reps}rep.csv")
    run(args.n_reps, args.seed_offset, args.test_size, args.low, args.high, out)
