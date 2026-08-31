"""H95前段: E2は11項目を「全部同じ扱い」にしているが、変えられないか。

現行のE2は生の1〜5をそのまま等間隔の数値として線形に読む。one-hot化・交互作用・
LightGBM化・H22閾値指標の本番投入は却下済み（`expert_architecture_audit_20260829.md`）
なので、ここではまだ試していない3方向を測る。

``encoding``
    項目ごとに水準の間隔を学習する。単調制約つき(isotonic)、自由（縮小つき水準平均）、
    項目ごとの段差指標の3種。one-hot は自由すぎ、線形は硬すぎるので、その中間。
    符号化はfold内でネスト交差fitするので、目的変数の漏れは無い。

``items``
    項目を絞る。弱い2項目を落とす／fold内でAUC上位k項目だけ使う。

``fill``
    `アンケート７` の未導入企業（356社）を満足度スケールのどこに置くか。
    `dataset.ZERO_FILL` は 0 に置いており、これは「非常に低い」(1) より下という意味に
    なる。`survey_features.py` の domain 値は 3.3。

いずれも E2 の出力だけを差し替えて、他7本はキャッシュのまま本番の外側foldで合成する
（`expert_architecture_audit_20260829.md` の「対象だけ差し替える」方式）。
E2 の再現はキャッシュと spearman 1.0000 で一致することを確認済み。

Run:
  .venv/bin/python exp/compare_survey_variants.py --mode encoding
  .venv/bin/python exp/compare_survey_variants.py --mode items
  .venv/bin/python exp/compare_survey_variants.py --mode fill
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS  # noqa: E402
from ensemble_experts import SURVEY_LR, THS, blend_oof  # noqa: E402

EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
ITEMS = [f"アンケート{n}" for n in "１２３４５６７８９"] + ["アンケート１０", "アンケート１１"]
A7 = "アンケート７"
WEAK = ("アンケート９", "アンケート１１")   # 単体AUCが 0.503 / 0.509 の2項目
FILLS = (0.0, 1.0, 2.0, 3.0, 3.3, 3.5, 4.0, 5.0)
NESTED_OFFSET = 777


def _matrix(frame, fill=0.0):
    """E2 の入力を再現する。末尾列は アンケート７_isna。"""
    X = frame[ITEMS].astype(float).copy()
    isna = frame[A7].isna()
    X.loc[isna, A7] = fill
    X["アンケート７_isna"] = isna.astype(float)
    return X.fillna(0).values


def _enc_iso(col_fit, y_fit, col_app):
    ir = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    return ir.fit(col_fit, y_fit).predict(col_app)


def _enc_level(col_fit, y_fit, col_app, k=20):
    stat = pd.Series(y_fit).groupby(pd.Series(col_fit)).agg(["mean", "size"])
    prior = y_fit.mean()
    shrunk = (stat["mean"] * stat["size"] + prior * k) / (stat["size"] + k)
    return pd.Series(col_app).map(shrunk).fillna(prior).values


def _enc_step(col_fit, y_fit, col_app):
    best, cut = 0.0, None
    for t in (1.5, 2.5, 3.5, 4.5):
        ind = (col_fit > t).astype(float)
        if 5 < ind.sum() < len(ind) - 5:
            gap = abs(roc_auc_score(y_fit, ind) - 0.5)
            if gap > best:
                best, cut = gap, t
    return (col_app > cut).astype(float) if cut is not None else np.zeros(len(col_app))


ENCODERS = {"iso": _enc_iso, "level": _enc_level, "step": _enc_step}


def _transform(kind, X_fit, y_fit, X_app):
    f = ENCODERS[kind]
    return np.column_stack([f(X_fit[:, j], y_fit, X_app[:, j])
                            for j in range(len(ITEMS))])


def _transform_nested(kind, X_fit, y_fit, seed):
    """学習側の符号化は内側交差で作る（同じ行で当てた符号化を見せない）。"""
    out = np.zeros((len(X_fit), len(ITEMS)))
    for a, b in StratifiedKFold(5, shuffle=True,
                                random_state=seed + NESTED_OFFSET).split(X_fit, y_fit):
        out[b] = _transform(kind, X_fit[a], y_fit[a], X_fit[b])
    return out


def _fit_e2(X_tr, X_te, y, seed, kind=None, keep=None, select_k=None):
    oof, test_parts = np.zeros(len(y)), []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_tr, y):
        cols = list(range(X_tr.shape[1]))
        if keep is not None:
            cols = keep + [X_tr.shape[1] - 1]
        if select_k is not None:
            gaps = [abs(roc_auc_score(y[a], X_tr[a, j]) - 0.5)
                    for j in range(len(ITEMS))]
            cols = list(np.argsort(gaps)[::-1][:select_k]) + [X_tr.shape[1] - 1]
        f_tr, f_va, f_te = X_tr[a][:, cols], X_tr[b][:, cols], X_te[:, cols]
        if kind is not None:
            f_tr = np.hstack([f_tr, _transform_nested(kind, X_tr[a], y[a], seed)])
            f_va = np.hstack([f_va, _transform(kind, X_tr[a], y[a], X_tr[b])])
            f_te = np.hstack([f_te, _transform(kind, X_tr[a], y[a], X_te)])
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(**SURVEY_LR)).fit(f_tr, y[a])
        oof[b] = model.predict_proba(f_va)[:, 1]
        test_parts.append(model.predict_proba(f_te)[:, 1])
    return oof, np.mean(test_parts, axis=0)


def _score(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS)}


def _run(arms, seeds, title, out_csv, base_label):
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    rows = []
    for seed in seeds:
        cache = np.load(CONFIGS["h2h7both4"].format(seed=seed), allow_pickle=True)
        y = cache["y"]
        oof = {n: cache["oof_" + n] for n in EXPERTS}
        te = {n: cache["te_" + n] for n in EXPERTS}
        base, _, _, _ = blend_oof(y, oof, te, seed=seed, experts=EXPERTS)
        rows.append({"seed": seed, "arm": base_label,
                     "e2": roc_auc_score(y, oof["E2_survey"]), **_score(y, base)})
        for label, (fill, kw) in arms.items():
            X_tr, X_te = _matrix(train, fill), _matrix(test, fill)
            o, t = _fit_e2(X_tr, X_te, y, seed, **kw)
            cand, _, _, _ = blend_oof(y, {**oof, "E2_survey": o},
                                      {**te, "E2_survey": t}, seed=seed,
                                      experts=EXPERTS)
            rows.append({"seed": seed, "arm": label,
                         "e2": roc_auc_score(y, o), **_score(y, cand)})
    r = pd.DataFrame(rows)
    b = r[r.arm == base_label].sort_values("seed").reset_index(drop=True)
    print(f"=== {title}（{len(seeds)}seed、本番の外側fold）===\n")
    print(f"{'E2の作り':<26}{'E2単体AUC':>11}{'ΔAUC':>10}{'ΔAP':>10}{'ΔF1':>10}"
          f"{'ΔAP>0':>8}  判定")
    print(f"{base_label:<26}{b.e2.mean():>11.4f}")
    for label in arms:
        c = r[r.arm == label].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{label:<26}{c.e2.mean():>11.4f}{(c.auc - b.auc).mean():>+10.4f}"
              f"{(c.ap - b.ap).mean():>+10.4f}{(c.f1 - b.f1).mean():>+10.4f}"
              f"{int((c.ap > b.ap).sum()):>6}/{len(c)}  {v}")
    r.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="encoding",
                    choices=("encoding", "items", "fill"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    if args.mode == "encoding":
        _run({"単調な水準符号化": (0.0, {"kind": "iso"}),
              "自由な水準符号化(縮小)": (0.0, {"kind": "level"}),
              "項目ごとの段差指標": (0.0, {"kind": "step"})},
             args.seeds, "水準の間隔を項目ごとに学習する",
             "exp/survey_level_encoding.csv", "現行(生値の線形)")
    elif args.mode == "items":
        keep = [j for j, c in enumerate(ITEMS) if c not in WEAK]
        _run({"弱い2項目(Q9,Q11)を落とす": (0.0, {"keep": keep}),
              "fold内でAUC上位5項目": (0.0, {"select_k": 5}),
              "fold内でAUC上位8項目": (0.0, {"select_k": 8})},
             args.seeds, "項目を絞る", "exp/survey_item_selection.csv",
             "現行(11項目すべて)")
    else:
        _run({f"a7未導入を{f:g}に置く": (f, {}) for f in FILLS if f != 0.0},
             args.seeds, "a7(既存ツール満足度)の未導入企業をどこに置くか",
             "exp/survey_q7_fill.csv", "現行(0=『非常に低い』より下)")


if __name__ == "__main__":
    main()
