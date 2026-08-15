"""H27 E7(不満×財務)の事前診断 — 相関ゲートとモデル族の選択。

事前登録した停止条件（`hypothesis_log.md` 2026-08-14）のうち 1〜3 をここで判定する。
合成の確定判断は `holdout_check.py --cross` で行う。

  1. 相関ゲート: E7 の OOF と既存6本いずれかの |Spearman| >= 0.90 なら REJECT
  2. 単体AP: 正例率 0.2412 を明確に上回ること
  3. モデル族: lgbm / lr を同一の外側5分割で比較し AP の良い方を採る

既存6本の OOF は `_experts_seed{seed}.npz`（exp019 構成）から読む。E7 は
`ensemble_experts.compute_expert_preds` と同一の分割・同一の列で作り直すので、
ここで得た OOF はそのまま合成に使えるものと一致する。

  python exp/diagnose_cross.py [--seed 42]

出力: exp/_cross_diag_scores.csv / exp/_cross_diag_corr.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import (  # noqa: E402
    EXPERTS, OUT_DIR, THS, build_features, fit_e7,
)
from expert_groups import e7_cross_cols  # noqa: E402

CORR_GATE = 0.90  # exp021 が 0.927 で却下、exp026 が 0.560 で通過した線


def _scores(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=max(f1s))


def e7_oof(seed, e7_model):
    """E7 の OOF と test 予測を、compute_expert_preds と同一の分割で作る。"""
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    c7 = e7_cross_cols()
    oof, te = np.zeros(len(X)), []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        p, q = fit_e7(X.iloc[tr][c7], y[tr], X.iloc[va][c7], y[va],
                      Xte[c7], seed, e7_model)
        oof[va] = p
        te.append(q)
    return y, oof, np.mean(te, axis=0)


def run(seed):
    base_path = os.path.join(OUT_DIR, f"_experts_seed{seed}.npz")
    d = np.load(base_path)
    y = d["y"]
    base = {n: d[f"oof_{n}"] for n in EXPERTS}
    rate = float(y.mean())
    print(f"土台: {base_path}（exp019 構成6本） / 正例率 {rate:.4f}")

    rows, oofs = {}, {}
    for n in EXPERTS:
        rows[n] = _scores(y, base[n])
    for m in ("lgbm", "lr"):
        print(f"\nE7({m}) を5分割で学習中 ...", flush=True)
        _, o, t = e7_oof(seed, m)
        oofs[m] = (o, t)
        rows[f"E7_cross({m})"] = _scores(y, o)

    tbl = pd.DataFrame(rows).T
    tbl["AP倍"] = tbl["ap"] / rate
    print(f"\n=== 単体スコア (seed={seed}) ===")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv(os.path.join(OUT_DIR, "_cross_diag_scores.csv"))

    pick = max(("lgbm", "lr"), key=lambda m: tbl.loc[f"E7_cross({m})", "ap"])
    d_ap = tbl.loc["E7_cross(lgbm)", "ap"] - tbl.loc["E7_cross(lr)", "ap"]
    print(f"\n[条件3] モデル族: **{pick}** を採用 "
          f"(lgbm - lr の AP {d_ap:+.4f})")
    print(f"[条件2] 単体AP {tbl.loc[f'E7_cross({pick})', 'ap']:.4f} "
          f"= 正例率の {tbl.loc[f'E7_cross({pick})', 'AP倍']:.2f}倍")

    print(f"\n=== E7 と既存6本の Spearman相関 (ゲート {CORR_GATE}) ===")
    corr = {}
    for m in ("lgbm", "lr"):
        corr[m] = {n: float(spearmanr(oofs[m][0], base[n]).statistic)
                   for n in EXPERTS}
    cdf = pd.DataFrame(corr).T
    cdf["最大|r|"] = cdf.abs().max(axis=1)
    print(cdf.to_string(float_format=lambda v: f"{v:.3f}"))
    cdf.to_csv(os.path.join(OUT_DIR, "_cross_diag_corr.csv"))

    mx = cdf.loc[pick, "最大|r|"]
    print(f"\n[条件1] 採用モデル({pick})の最大|r| = {mx:.3f} → "
          + ("**REJECT（相関ゲート違反）**" if mx >= CORR_GATE
             else f"通過（{CORR_GATE} 未満）"))
    if mx < CORR_GATE:
        print("\n次: python exp/holdout_check.py --n-reps 5 --cross"
              + ("" if pick == "lgbm" else " --e7-model lr"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args().seed)
