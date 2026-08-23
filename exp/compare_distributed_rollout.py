"""H50 多拠点DX教育展開負荷8列LRをexp032へ追加して比較する。"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_buying_center import (  # noqa: E402
    BASE_MEMBERS, FIXED_ALPHAS, OUT_DIR, _cross_lr, _scores, blend_oof,
    load_base_predictions,
)
from decision import format_report, verdict  # noqa: E402
from distributed_rollout_features import make_distributed_rollout_features  # noqa: E402

NAME = "E16_distributed_rollout"


def make_candidate_oof(seed, cache=True):
    path = os.path.join(OUT_DIR, f"_h50_distributed_rollout_seed{seed}.npz")
    if cache and os.path.exists(path):
        data = np.load(path)
        return data["y"], data["oof"], data["test"]
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X = make_distributed_rollout_features(train)
    Xte = make_distributed_rollout_features(test)
    y = train["購入フラグ"].to_numpy(dtype=int)
    oof, test_parts = np.zeros(len(y)), []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(folds.split(X, y), 1):
        print(f"    H50 fold {k}/5 ...", flush=True)
        p, q = _cross_lr(X.iloc[tr], y[tr], X.iloc[va], Xte)
        oof[va], test_parts = p, test_parts + [q]
    test_pred = np.mean(test_parts, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def run(seeds):
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    trf = make_distributed_rollout_features(train)
    tef = make_distributed_rollout_features(test)
    stability = pd.DataFrame({
        "train_mean": trf.mean(), "test_mean": tef.mean(),
        "train_std": trf.std(), "test_std": tef.std(),
    })
    print("=== H50 特徴分布（ラベル非参照） ===")
    print(stability.to_string(float_format=lambda x: f"{x:.3f}"))
    stability.to_csv(os.path.join(OUT_DIR, "_h50_feature_stability.csv"))

    rows, weights, correlations, payloads = [], [], [], []
    for position, seed in enumerate(seeds, 1):
        print(f"\n=== seed {seed} ({position}/{len(seeds)}) ===", flush=True)
        y, oof, te = load_base_predictions(seed)
        yy, candidate_oof, candidate_test = make_candidate_oof(seed)
        assert np.array_equal(y, yy)
        correlations.append(pd.Series({
            name: spearmanr(candidate_oof, oof[name]).statistic
            for name in BASE_MEMBERS
        }, name=seed))
        all_oof, all_te = {**oof, NAME: candidate_oof}, {**te, NAME: candidate_test}
        payloads.append((seed, y, all_oof, all_te))
        base, _, _, _ = blend_oof(y, all_oof, all_te, seed, None,
                                  experts=BASE_MEMBERS)
        candidate, _, w, _ = blend_oof(
            y, all_oof, all_te, seed, None, experts=BASE_MEMBERS + (NAME,),
        )
        rows.extend([
            dict(seed=seed, variant="exp032", **_scores(y, base)),
            dict(seed=seed, variant="exp032+H50", **_scores(y, candidate)),
            dict(seed=seed, variant="H50単体", **_scores(y, candidate_oof)),
        ])
        weights.append(pd.Series(w, index=BASE_MEMBERS + (NAME,), name=seed))

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT_DIR, "_h50_5seed_scores.csv"), index=False)
    base = scores[scores.variant == "exp032"].sort_values("seed").reset_index(drop=True)
    candidate = scores[scores.variant == "exp032+H50"].sort_values("seed").reset_index(drop=True)
    base.attrs["n_feat"], candidate.attrs["n_feat"] = 8, 9
    result, deltas = verdict(base, candidate)
    print("\n" + format_report(
        "H50", "多拠点DX教育展開負荷8列LR Expertをexp032へ追加",
        base, candidate, result, deltas,
    ))

    corr = pd.DataFrame(correlations)
    corr.to_csv(os.path.join(OUT_DIR, "_h50_5seed_corr.csv"))
    print("\n=== H50と既存ExpertのSpearman（seed平均） ===")
    print(corr.mean().sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
    W = pd.DataFrame(weights).div(pd.DataFrame(weights).sum(axis=1), axis=0) * 100
    W.to_csv(os.path.join(OUT_DIR, "_h50_5seed_weights.csv"))
    print(f"\nH50メタ重み: {W[NAME].mean():.2f}% ± {W[NAME].std():.2f}%")

    fixed_rows = []
    for alpha in FIXED_ALPHAS:
        for seed, y, oof, te in payloads:
            bp, _, _, _ = blend_oof(y, oof, te, seed, alpha, experts=BASE_MEMBERS)
            cp, _, w, _ = blend_oof(y, oof, te, seed, alpha,
                                    experts=BASE_MEMBERS + (NAME,))
            sb, sc = _scores(y, bp), _scores(y, cp)
            fixed_rows.append(dict(
                alpha=alpha, seed=seed, d_auc=sc["auc"] - sb["auc"],
                d_ap=sc["ap"] - sb["ap"], d_f1=sc["f1"] - sb["f1"],
                weight_pct=100 * w[-1] / (w.sum() or 1),
            ))
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(os.path.join(OUT_DIR, "_h50_fixed_alpha_scores.csv"), index=False)
    print("\n=== 固定alpha感度 ===")
    for alpha, data in fixed.groupby("alpha"):
        print(f"alpha={alpha:.3f}: dAP={data.d_ap.mean():+.4f} "
              f"({int((data.d_ap > 0).sum())}/{len(data)}), "
              f"dAUC={data.d_auc.mean():+.4f} "
              f"({int((data.d_auc > 0).sum())}/{len(data)}), "
              f"dF1={data.d_f1.mean():+.4f}, weight={data.weight_pct.mean():.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    run(parser.parse_args().seeds)
