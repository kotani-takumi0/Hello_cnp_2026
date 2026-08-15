"""現本命exp032のパイプライン全体をCV seed平均する。

各seedで8エキスパートと非負メタブレンドを独立に作り、最終的なOOF/test確率を
単純平均する。候補特徴を増やさず、fold分割・学習乱数による分散だけを下げる。

  python3 exp/ensemble_multiseed_best.py --seeds 42 0 1 2 3 [--submit]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import (  # noqa: E402
    _scores, blend_oof, compute_expert_preds, expert_names,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERTS = expert_names(cross=True, linear=True)


def run(seeds, submit=False):
    oof_parts, test_parts, rows, weights = [], [], [], []
    y_ref = None
    for seed in seeds:
        print(f"--- seed {seed} ---", flush=True)
        y, oof, test = compute_expert_preds(
            seed, concat_embed=True, cross=True, e7_model="lr",
            linear=True, linear_c=0.03,
        )
        if y_ref is None:
            y_ref = y
        else:
            assert np.array_equal(y_ref, y), "seed間で目的変数の並びが違う"
        p, q, w, alphas = blend_oof(
            y, oof, test, seed, alpha=None, experts=EXPERTS,
        )
        score = _scores(y, p)
        rows.append(dict(seed=seed, **score))
        oof_parts.append(p)
        test_parts.append(q)
        weights.append(w)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in score.items())
              + f"  alpha={alphas}", flush=True)

    oof_parts = np.asarray(oof_parts)
    test_parts = np.asarray(test_parts)
    mean_oof = oof_parts.mean(axis=0)
    mean_test = test_parts.mean(axis=0)
    mean_score = _scores(y_ref, mean_oof)

    frame = pd.DataFrame(rows)
    print("\n=== seed別OOF ===")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nseed別指標の平均±std:")
    for key in ("auc", "ap", "f1", "th"):
        print(f"  {key.upper():4s} {frame[key].mean():.4f} ± {frame[key].std():.4f}")
    print("\n=== 5seed確率平均OOF ===")
    print("  " + "  ".join(f"{k.upper()} {v:.4f}" for k, v in mean_score.items()))
    print(f"予測正例率: OOF={float((mean_oof >= mean_score['th']).mean()):.4f} "
          f"test={float((mean_test >= mean_score['th']).mean()):.4f}")

    if len(seeds) > 1:
        corr = spearmanr(oof_parts.T).correlation
        upper = corr[np.triu_indices(len(seeds), 1)]
        print(f"seed間OOF順位相関: min={upper.min():.4f} mean={upper.mean():.4f}")
    test_std = test_parts.std(axis=0)
    print(f"test確率のseed間std: mean={test_std.mean():.5f} "
          f"p95={np.quantile(test_std, .95):.5f} max={test_std.max():.5f}")

    W = pd.DataFrame(np.asarray(weights), index=seeds, columns=EXPERTS)
    W = W.div(W.sum(axis=1), axis=0) * 100
    print("\n=== メタ重み（seed別、正規化%）===")
    print(W.to_string(float_format=lambda v: f"{v:.1f}"))

    tag = f"exp032_multiseed{len(seeds)}_" + "-".join(map(str, seeds))
    artifact = os.path.join(ROOT, "exp", f"_{tag}.npz")
    np.savez(artifact, seeds=np.asarray(seeds), y=y_ref, oof=oof_parts,
             test=test_parts, mean_oof=mean_oof, mean_test=mean_test,
             weights=np.asarray(weights))
    print(f"保存: {artifact}")

    if submit:
        threshold = mean_score["th"]
        label = (mean_test >= threshold).astype(int)
        test = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
        sample = pd.read_csv(os.path.join(ROOT, "data", "sample_submit.csv"),
                             header=None, names=["企業ID", "購入フラグ"])
        pred = pd.DataFrame({"企業ID": test["企業ID"].values, "pred": label})
        out = sample[["企業ID"]].merge(pred, on="企業ID", how="left")
        assert out["pred"].notna().all()
        path = os.path.join(ROOT, "submission", f"submission_{tag}.csv")
        out.assign(pred=out["pred"].astype(int)).to_csv(
            path, index=False, header=False, lineterminator="\n")
        print(f"保存: {path} 正例={int(label.sum())} (th={threshold:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    run(args.seeds, args.submit)
