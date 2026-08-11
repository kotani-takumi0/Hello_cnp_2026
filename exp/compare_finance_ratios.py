"""H21 財務比の足切り診断と、E1単体での10seedペア比較。

過去の H2/H3/H7 は E0(94列) の上で検証して全滅した。E0 では
`DX展望_購入確率` が gain 単独首位（2位の2.3倍）で、比の寄与が埋もれる。
一方 E1(32列) では H1利益率3本だけで gain 23% を取っているので、
同じ「比」でも寄与の大きさが違う。ここは E1 の中だけで判定する。

  python exp/compare_finance_ratios.py [--mode diag|compare] [--n-seeds 10]

diag    : 候補の単体AUC/AP（正例率0.241をどれだけ上回るか）と既存E1列との相関
compare : E1単体の10seedペア比較。判定は decision.py の基準をそのまま使う
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import BASE_PARAMS, TARGET, preprocess  # noqa: E402
from make_submission_h1_h8 import add_h1  # noqa: E402
from expert_groups import e1_finance_cols  # noqa: E402
from finance_ratio_features import (  # noqa: E402
    FINANCE_RATIO_COLS, _safe_div, add_finance_ratios,
)
from decision import format_report, verdict  # noqa: E402

THS = np.arange(0.05, 0.95, 0.005)


def _rejected_ratios(t):
    """過去に却下された比。診断で「なぜ却下されたか」を再確認するために残す。"""
    emp = t["従業員数"].replace(0, np.nan)
    return {
        # H2 財務健全性
        "自己資本比率": _safe_div(t["自己資本"], t["総資産"]),
        "負債比率": _safe_div(t["負債"], t["自己資本"]),
        "ROA": _safe_div(t["当期純利益"], t["総資産"]),
        "ROE": _safe_div(t["当期純利益"], t["自己資本"]),
        # H7 CF比率
        "営業CFマージン": _safe_div(t["営業CF"], t["売上"]),
        "FCF比率": _safe_div(t["営業CF"] + t["投資CF"], t["総資産"]),
        "現金ROA": _safe_div(t["営業CF"], t["総資産"]),
        # H3 1人あたり
        "売上_per従業員": _safe_div(t["売上"], emp),
        "総資産_per従業員": _safe_div(t["総資産"], emp),
        "ソフト投資_per従業員": _safe_div(
            t["無形固定資産変動(ソフトウェア関連)"], emp),
    }


def _uni(y, v):
    """単体シグナル。欠損は除いて測り、向きは揃える（負相関でも効けば効く）。"""
    v = pd.Series(v).astype(float)
    m = v.notna().values
    if m.sum() < 30 or len(np.unique(v[m])) < 2:
        return dict(auc=np.nan, ap=np.nan, cov=m.mean())
    yy, vv = y[m], v[m].values
    a = roc_auc_score(yy, vv)
    return dict(auc=max(a, 1 - a), cov=m.mean(),
                ap=max(average_precision_score(yy, vv),
                       average_precision_score(yy, -vv)))


def _load():
    train = pd.read_csv("data/train.csv")
    tp = preprocess(train)
    y = tp[TARGET].values.astype(int)
    X = add_h1(train, tp.drop(columns=[TARGET]))
    return train, X, y


def run_diag():
    train, X, y = _load()
    base_rate = y.mean()
    c1 = e1_finance_cols()
    Xr = add_finance_ratios(train, X)
    cand = {c: Xr[c] for c in FINANCE_RATIO_COLS}
    cand.update(_rejected_ratios(train))

    print(f"train={len(y)}行 正例率={base_rate:.4f}\n")
    print("=== E1既存32列の単体シグナル（AP上位10） ===")
    t0 = pd.DataFrame({c: _uni(y, X[c]) for c in c1}).T.sort_values(
        "ap", ascending=False)
    t0["ap倍率"] = t0["ap"] / base_rate
    print(t0.head(10).to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== 候補の単体シグナル + 既存E1列との最大|Spearman| ===")
    rows = {}
    for c, v in cand.items():
        r = X[c1].corrwith(pd.Series(v).astype(float), method="spearman").abs()
        rows[c] = dict(_uni(y, v), ap倍率=_uni(y, v)["ap"] / base_rate,
                       max_r=r.max(), against=r.idxmax(),
                       採用="○" if c in FINANCE_RATIO_COLS else "")
    print(pd.DataFrame(rows).T.sort_values("ap", ascending=False).to_string())
    print("\n採用基準: 単体APが正例率を明確に上回る / 既存E1列との|r|<0.5 /"
          " 候補どうしでも重複しない")
    print("  例: 自己資本比率(|r|=0.498)は基準内だが 純資産_総資産比 と r=0.992"
          "（自己資本と純資産が r=0.999）なので片方だけ採る")


def _run_seed(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        m = lgb.train({**BASE_PARAMS, "seed": seed},
                      lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    return dict(auc=roc_auc_score(y, oof), ap=average_precision_score(y, oof),
                f1=max(f1_score(y, (oof >= t).astype(int)) for t in THS))


def _table(X, y, n_seeds):
    df = pd.DataFrame([_run_seed(X, y, s) for s in range(n_seeds)])
    df.attrs["n_feat"] = X.shape[1]
    return df


def run_compare(n_seeds):
    train, X, y = _load()
    Xr = add_finance_ratios(train, X)
    base_X = X[e1_finance_cols()]
    variants = {
        "H21_採用4本": e1_finance_cols(with_ratios=True),
        "H2_参考(却下済み)": e1_finance_cols() + ["自己資本比率", "負債比率",
                                                  "ROA", "ROE"],
    }
    for c in ("自己資本比率", "負債比率", "ROA", "ROE"):
        Xr[c] = _rejected_ratios(train)[c].values

    print(f"baseline: E1 {base_X.shape[1]}列 / {n_seeds}seed", flush=True)
    base = _table(base_X, y, n_seeds)
    for name, cols in variants.items():
        print(f"\n候補 {name} ...", flush=True)
        cand = _table(Xr[cols], y, n_seeds)
        v, d = verdict(base, cand)
        print(format_report(name, "E1に比を追加", base, cand, v, d))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["diag", "compare"], default="compare")
    p.add_argument("--n-seeds", type=int, default=10)
    a = p.parse_args()
    run_diag() if a.mode == "diag" else run_compare(a.n_seeds)
