"""エキスパートの構成員を絞ったときの増減を、同一seedペアで比較する。

5seed の重み推定で E3(DX展望テキスト) 2.2%±1.8 / E4(組織図テキスト) 4.2%±3.1 と、
**σが平均と同じ大きさ＝重みが定まっていない**構成員が2本あった。742行に対して
構成員が多すぎる可能性があるので、落として悪化しないかを見る。

E3/E4 の情報は E0アンカーの中にテキストスタッキング確率として既に入っているため、
独立メンバーとしては冗長になっている、というのが仮説。

キャッシュ(`_experts_seed{seed}.npz`)には常に全6本の予測が入っているので、
部分集合の評価に再計算は不要。

  python exp/compare_expert_subsets.py [--seeds 42 0 1 2 3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import EXPERTS, _scores, blend_oof, compute_expert_preds  # noqa: E402

METRICS = ("auc", "ap", "f1")
SUBSETS = {
    "6本(全部)": EXPERTS,
    "5本(-E3)": tuple(n for n in EXPERTS if n != "E3_dx_text"),
    "5本(-E4)": tuple(n for n in EXPERTS if n != "E4_org_text"),
    "4本(-E3,-E4)": tuple(n for n in EXPERTS if n not in ("E3_dx_text", "E4_org_text")),
    "3本(-E3,-E4,-E6)": ("E0_anchor", "E1_finance", "E2_survey"),
}


def run(seeds, alpha):
    rows, weights = [], {}
    for seed in seeds:
        y, oof, te = compute_expert_preds(seed)
        rows.append(dict(seed=seed, subset="E0単体", **_scores(y, oof["E0_anchor"])))
        for label, members in SUBSETS.items():
            blend, _, w, _ = blend_oof(y, oof, te, seed, alpha, experts=members)
            rows.append(dict(seed=seed, subset=label, **_scores(y, blend)))
            weights.setdefault(label, []).append(
                pd.Series(100 * w / (w.sum() or 1), index=list(members)))

    df = pd.DataFrame(rows)
    order = ["E0単体"] + list(SUBSETS)
    print(f"=== OOFスコア (n_seeds={len(seeds)}, alpha={'auto' if alpha is None else alpha}) ===")
    g = df.groupby("subset")[list(METRICS)].agg(["mean", "std"]).reindex(order)
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    piv = df.pivot(index="seed", columns="subset")
    print("\n=== 6本(全部) との同一seedペア差分 ===")
    for label in order:
        if label == "6本(全部)":
            continue
        line = [f"  {label:18s}"]
        for k in METRICS:
            d = (piv[(k, label)] - piv[(k, "6本(全部)")]).reindex(seeds)
            line.append(f"Δ{k.upper()} {d.mean():+.4f} (正{int((d > 0).sum())}/{len(d)})")
        print("  ".join(line))

    print("\n=== 4本構成の重み (正規化%, seed別) ===")
    W = pd.DataFrame(weights["4本(-E3,-E4)"], index=seeds)
    W.index.name = "seed"
    print(W.to_string(float_format=lambda v: f"{v:.1f}"))
    print("  平均±std: " + " / ".join(
        f"{c} {W[c].mean():.1f}±{W[c].std():.1f}" for c in W.columns))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    p.add_argument("--alpha", type=float, default=None)
    a = p.parse_args()
    run(a.seeds, a.alpha)
