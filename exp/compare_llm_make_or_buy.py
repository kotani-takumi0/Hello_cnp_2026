"""LLM Make-or-Buy特徴を既存8本へ独立LRとして追加するペア比較。

購入ラベルを渡さず抽出済みの ``data/llm_make_or_buy`` を利用する。
既存8本の入力・予測は変更せず、同じ外側foldで9本目のOOFを作る。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS, blend_oof  # noqa: E402
from llm_make_or_buy_features import PARENT_COLS, PRODUCT_COLS  # noqa: E402

BASE_EXPERTS = (
    "E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
    "E4_org_text", "E6_manual", "E7_cross", "E0b_linear",
)
LLM_NAME = "E10_llm_makebuy"

def score(y, p):
    f1 = max(f1_score(y, p >= t) for t in THS)
    return {"auc": roc_auc_score(y, p),
            "ap": average_precision_score(y, p), "f1": f1}


def load_llm_features(mode):
    root = "data/llm_make_or_buy"
    tr = pd.read_csv(f"{root}/train_make_or_buy_features.csv")
    te = pd.read_csv(f"{root}/test_make_or_buy_features.csv")
    raw_tr = pd.read_csv("data/train.csv")
    raw_te = pd.read_csv("data/test.csv")
    tr = raw_tr[["企業ID"]].merge(tr, on="企業ID", how="left",
                                  validate="one_to_one")
    te = raw_te[["企業ID"]].merge(te, on="企業ID", how="left",
                                  validate="one_to_one")
    cols = list(PARENT_COLS + (PRODUCT_COLS if mode == "products" else ()))
    # API失敗3件を通常のunknownと区別する。補完値は各foldのtrainだけから学ぶ。
    tr["llm_row_missing"] = tr[list(PARENT_COLS)].isna().all(axis=1).astype(int)
    te["llm_row_missing"] = te[list(PARENT_COLS)].isna().all(axis=1).astype(int)
    cols.append("llm_row_missing")
    return tr[cols], te[cols], cols


def llm_oof(seed, y, X, Xte):
    oof = np.zeros(len(X))
    test_parts = []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.1, solver="liblinear", max_iter=3000,
                               random_state=seed),
        )
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_parts.append(model.predict_proba(Xte)[:, 1])
    return oof, np.mean(test_parts, axis=0)


def main(seeds, mode, output, operation):
    X, Xte, cols = load_llm_features(mode)
    rows = []
    for seed in seeds:
        path = f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03.npz"
        d = np.load(path)
        y = d["y"]
        oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        p, q = llm_oof(seed, y, X, Xte)
        oof[LLM_NAME], te[LLM_NAME] = p, q
        base, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
        sb, sl = score(y, base), score(y, p)
        candidates = (None,) if operation == "add" else BASE_EXPERTS
        for replaced in candidates:
            names = (BASE_EXPERTS + (LLM_NAME,) if replaced is None else
                     tuple(LLM_NAME if n == replaced else n for n in BASE_EXPERTS))
            aug, _, weights, _ = blend_oof(y, oof, te, seed, experts=names)
            sa = score(y, aug)
            row = {"seed": seed, "replaced": replaced, "llm": sl,
                   "base": sb, "candidate": sa,
                   "delta": {k: sa[k] - sb[k] for k in sb},
                   "llm_weight": float(weights[names.index(LLM_NAME)])}
            rows.append(row)
            print(seed, "replace", replaced, "delta", row["delta"],
                  "weight", round(row["llm_weight"], 4))
    summary = {}
    labels = [None] if operation == "add" else list(BASE_EXPERTS)
    for replaced in labels:
        subset = [r for r in rows if r["replaced"] == replaced]
        key = "add" if replaced is None else replaced
        summary[key] = {}
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["delta"][metric] for r in subset])
            summary[key][metric] = {"mean": float(vals.mean()),
                                    "wins": int((vals > 0).sum()),
                                    "n": len(vals)}
    result = {"mode": mode, "operation": operation, "seeds": seeds,
              "features": cols, "rows": rows, "summary": summary}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
    p.add_argument("--mode", choices=["parents", "products"], default="parents")
    p.add_argument("--operation", choices=["add", "replace"], default="add")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    main(a.seeds, a.mode, a.output, a.operation)
