"""H28 E0b(94列 one-hot + LR)の事前診断 — 相関ゲートと C の選択。

事前登録した停止条件（`hypothesis_log.md` 2026-08-14）の 1〜3 をここで判定する。
確定判断は `holdout_check.py --n-reps 15 --concat-embed --cross --e7-model lr --linear`。

E0b は E0 と**同じ94列**（fold内テキストスコア込み）を使うので、E0 の行列を1回作る
過程で C を総当たりできる。既存7本の OOF は exp031 のキャッシュから読む。

  python exp/diagnose_linear.py [--seed 42]

出力: exp/_linear_diag_scores.csv / exp/_linear_diag_corr.csv
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
from organization_features import (  # noqa: E402
    ORG_SCORE_COL, fold_org_preds, load_org_text,
)
from text_features import TEXT_COL  # noqa: E402
from expert_groups import e0_anchor_cols  # noqa: E402
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, OUT_DIR, THS, _nested_text, build_features, expert_names,
    fit_e0b,
)

CORR_GATE = 0.90
C_GRID = (0.03, 0.1, 0.3, 1.0)


def _scores(y, p):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=max(f1_score(y, (p >= t).astype(int)) for t in THS))


def e0b_oof(seed, c_grid=C_GRID):
    """E0 と同一の分割・同一の94列で、C ごとの E0b OOF を作る。"""
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    txt = train[TEXT_COL].fillna("").astype(str).values
    txt_te = test[TEXT_COL].fillna("").astype(str).values
    org, org_te = load_org_text(train), load_org_text(test)
    c0 = e0_anchor_cols(X.columns)

    oof = {c: np.zeros(len(X)) for c in c_grid}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"  fold {k}/5 ...", flush=True)
        t_tr, t_va, t_te = _nested_text(txt[tr], y[tr], txt[va], txt_te)
        o_tr, o_va, o_te = fold_org_preds(org[tr], y[tr], org[va], org_te)
        A_tr, A_va, A_te = X.iloc[tr][c0].copy(), X.iloc[va][c0].copy(), Xte[c0].copy()
        for col, v in ((DX_SCORE_COL, (t_tr, t_va, t_te)),
                       (ORG_SCORE_COL, (o_tr, o_va, o_te))):
            A_tr[col], A_va[col], A_te[col] = v
        for c in c_grid:
            oof[c][va], _ = fit_e0b(A_tr, y[tr], A_va, A_te, c)
    return y, oof


def run(seed):
    path = os.path.join(OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr.npz")
    d = np.load(path)
    names = expert_names(cross=True)
    base = {n: d[f"oof_{n}"] for n in names}
    y = d["y"]
    rate = float(y.mean())
    print(f"土台: {path}（exp031 構成7本） / 正例率 {rate:.4f}\n")

    _, oof = e0b_oof(seed)
    rows = {n: _scores(y, base[n]) for n in names}
    for c in C_GRID:
        rows[f"E0b_linear(C={c})"] = _scores(y, oof[c])
    tbl = pd.DataFrame(rows).T
    tbl["AP倍"] = tbl["ap"] / rate
    print(f"\n=== 単体スコア (seed={seed}) ===")
    print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv(os.path.join(OUT_DIR, "_linear_diag_scores.csv"))

    pick = max(C_GRID, key=lambda c: tbl.loc[f"E0b_linear(C={c})", "ap"])
    print(f"\n[条件3] C = **{pick}** を採用（単体APで選択）")
    print(f"[条件2] 単体AP {tbl.loc[f'E0b_linear(C={pick})', 'ap']:.4f} "
          f"= 正例率の {tbl.loc[f'E0b_linear(C={pick})', 'AP倍']:.2f}倍 "
          f"（E0アンカーは {tbl.loc['E0_anchor', 'ap']:.4f}）")

    print(f"\n=== E0b と既存7本の Spearman相関 (ゲート {CORR_GATE}) ===")
    corr = pd.DataFrame({
        f"C={c}": {n: float(spearmanr(oof[c], base[n]).statistic) for n in names}
        for c in C_GRID}).T
    corr["最大|r|"] = corr.abs().max(axis=1)
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))
    corr.to_csv(os.path.join(OUT_DIR, "_linear_diag_corr.csv"))

    mx = corr.loc[f"C={pick}", "最大|r|"]
    print(f"\n[条件1] 採用C({pick})の最大|r| = {mx:.3f} → "
          + ("**REJECT（相関ゲート違反）**" if mx >= CORR_GATE
             else f"通過（{CORR_GATE} 未満）"))
    if mx < CORR_GATE:
        print("\n次: python exp/holdout_check.py --n-reps 15 --concat-embed "
              f"--cross --e7-model lr --linear --linear-c {pick}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args().seed)
