"""exp018（エキスパート合成）を複数seedで確認する。

単seed(42)の結果だけでは採否を決めないという方針に従い、seedを振って
  - E0単体 / 等分ブレンド / メタ の同一seedペア差分
  - 学習された重み構成の安定性（特にE2アンケートが最大重みを取り続けるか）
を見る。エキスパートOOFは seed ごとに `_experts_seed{seed}.npz` にキャッシュ
されるので、2回目以降のalpha変更は数秒で再評価できる。

  python exp/ensemble_multiseed.py --seeds 42 0 1 2 3 [--alpha 0.01]
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
from meta_blend import to_logit, uniform_blend  # noqa: E402

METRICS = ("auc", "ap", "f1")


def run(seeds, alpha, survey_step=False, out_name="_ensemble_multiseed.csv"):
    rows, weights = [], {}
    for seed in seeds:
        print(f"--- seed {seed} ---", flush=True)
        y, oof, te = compute_expert_preds(seed, survey_step=survey_step)
        Z = to_logit(np.column_stack([oof[n] for n in EXPERTS]))
        blend, _, w, alphas = blend_oof(y, oof, te, seed, alpha)
        weights[seed] = w
        if alpha is None:
            print(f"  選ばれたalpha: {alphas}")
        rows.append(dict(seed=seed, model="E0_anchor", **_scores(y, oof["E0_anchor"])))
        rows.append(dict(seed=seed, model="uniform", **_scores(y, uniform_blend(Z))))
        rows.append(dict(seed=seed, model="meta", **_scores(y, blend)))

    df = pd.DataFrame(rows)
    print(f"\n=== OOFスコア (n_seeds={len(seeds)}, alpha={alpha}) ===")
    print(df.groupby("model")[list(METRICS)].agg(["mean", "std"])
          .to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== E0単体との同一seedペア差分 ===")
    piv = df.pivot(index="seed", columns="model")
    for k in METRICS:
        for m in ("uniform", "meta"):
            d = (piv[(k, m)] - piv[(k, "E0_anchor")]).reindex(seeds)
            print(f"  Δ{k.upper():3s} {m:8s}: mean {d.mean():+.4f}  std {d.std():.4f}  "
                  f"正 {int((d > 0).sum())}/{len(d)}  {np.round(d.values, 4)}")

    print("\n=== 学習された重み (正規化%, seed別) ===")
    W = pd.DataFrame({s: 100 * w / (w.sum() or 1) for s, w in weights.items()},
                     index=list(EXPERTS)).T
    W.index.name = "seed"
    print(W.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n  平均±std:")
    for n in EXPERTS:
        print(f"    {n:12s} {W[n].mean():5.1f}% ± {W[n].std():.1f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    df.to_csv(out, index=False)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    p.add_argument("--alpha", type=float, default=None,
                   help="省略時は外側trainの中だけで自動選択")
    p.add_argument("--survey-step", action="store_true",
                   help="E2 に H22 閾値指標4本を足す（キャッシュは別タグ）")
    a = p.parse_args()
    run(a.seeds, a.alpha, a.survey_step,
        "_ensemble_multiseed_step.csv" if a.survey_step
        else "_ensemble_multiseed.csv")
