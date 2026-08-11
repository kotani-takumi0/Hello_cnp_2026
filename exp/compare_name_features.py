"""企業名の低自由度特徴を E6 に混ぜたときの増減を、同一seedペアで測る。

EDAレポート14.4/16(優先度B)の方針に従い、**一度に全部入れず1本ずつ**足して差を見る。

E6(辞書・構造 48列)だけを作り直し、E0/E1/E2 はキャッシュから再利用する。
E6 は LightGBM 1本なので数秒で終わり、重いテキストスタッキングは触らない。

  python exp/compare_name_features.py [--seeds 42 0 1 2 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import (  # noqa: E402
    EXPERTS, _lgbm, _scores, blend_oof, build_features, compute_expert_preds,
)
from expert_groups import e6_manual_cols  # noqa: E402
from company_name_features import VARIANTS, add_company_name_features  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

BASE = tuple(n for n in EXPERTS if n not in ("E3_dx_text", "E4_org_text"))
METRICS = ("auc", "ap", "f1")


def e6_variant(X, Xte, y, seed, extra_cols):
    """E6 に extra_cols を足した版の OOF/test 予測を返す。"""
    cols = e6_manual_cols(X.columns) + list(extra_cols)
    oof, te = np.zeros(len(y)), []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p, q = _lgbm(X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va],
                     Xte[cols], seed)
        oof[va] = p
        te.append(q)
    return oof, np.mean(te, axis=0)


def run(seeds):
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X = add_company_name_features(train, X)
    Xte = add_company_name_features(test, Xte)

    rows = {}
    for seed in seeds:
        _, oof, te = compute_expert_preds(seed)
        # baseline: キャッシュ済みE6をそのまま使う
        b, _, _, _ = blend_oof(y, oof, te, seed, None, experts=BASE)
        rows.setdefault("baseline(企業名なし)", []).append(_scores(y, b))
        for label, extra in VARIANTS.items():
            o6, t6 = e6_variant(X, Xte, y, seed, extra)
            oof2, te2 = {**oof, "E6_manual": o6}, {**te, "E6_manual": t6}
            b, _, w, _ = blend_oof(y, oof2, te2, seed, None, experts=BASE)
            rows.setdefault(label, []).append(_scores(y, b))

    order = ["baseline(企業名なし)"] + list(VARIANTS)
    D = {k: pd.DataFrame(v) for k, v in rows.items()}
    print(f"=== OOFスコア (n_seeds={len(seeds)}) ===")
    print(f"{'構成':22s} {'AUC':>7s} {'AP':>7s} {'F1':>7s}")
    for k in order:
        d = D[k]
        print(f"{k:22s} {d.auc.mean():7.4f} {d.ap.mean():7.4f} {d.f1.mean():7.4f}")

    print("\n=== baseline との同一seedペア差分 ===")
    base = D["baseline(企業名なし)"]
    for k in list(VARIANTS):
        parts = []
        for m in METRICS:
            d = D[k][m].values - base[m].values
            parts.append(f"Δ{m.upper():3s} {d.mean():+.4f} (正{int((d > 0).sum())}/{len(d)})")
        print(f"  {k:22s} " + "  ".join(parts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    run(p.parse_args().seeds)
