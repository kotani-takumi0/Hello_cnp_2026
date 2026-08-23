"""H47 購買主体の完成度をexp032へ追加して全742行・複数seedで比較する。

実行例:
  .venv/bin/python exp/compare_buying_center.py --seeds 42 0 1 2 3

特徴・辞書は buying_center_features.py で事前固定。候補は7列だけのLR Expertとし、
現行exp032の8 Expertを変更せず9本目として追加する。
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from buying_center_features import (  # noqa: E402
    BUYING_CENTER_COLS, add_buying_center_features, make_buying_center_features,
)
from decision import format_report, verdict  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha, to_logit  # noqa: E402

NAME = "E13_buying_center"
BASE_MEMBERS = (
    "E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
    "E4_org_text", "E6_manual", "E7_cross", "E0b_linear",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_ALPHAS = (0.001, 0.003, 0.01, 0.03)
THRESHOLDS = np.arange(0.05, 0.95, 0.005)


def _load_features():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X = add_buying_center_features(train, pd.DataFrame(index=train.index))
    Xte = add_buying_center_features(test, pd.DataFrame(index=test.index))
    return train, test, X, train["購入フラグ"].to_numpy(dtype=int), Xte


def _cross_lr(Xtr, ytr, Xva, Xte):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, random_state=0,
    ))
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def _scores(y, p):
    f1s = [f1_score(y, p >= threshold) for threshold in THRESHOLDS]
    best = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1s[best], th=THRESHOLDS[best])


def blend_oof(y, oof, te, seed, alpha=None, experts=BASE_MEMBERS):
    z = to_logit(np.column_stack([oof[name] for name in experts]))
    zte = to_logit(np.column_stack([te[name] for name in experts]))
    out, weights, test_parts = np.zeros(len(y)), [], []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in folds.split(z, y):
        selected = alpha if alpha is not None else select_alpha(
            z[tr], y[tr], seed=seed, target="zero",
        )[0]
        w, intercept = fit_meta(z[tr], y[tr], alpha=selected, target="zero")
        out[va] = apply_meta(z[va], w, intercept)
        weights.append(w)
        test_parts.append(apply_meta(zte, w, intercept))
    return out, np.mean(test_parts, axis=0), np.mean(weights, axis=0), None


def load_base_predictions(seed):
    path = os.path.join(
        OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr_lin0.03.npz",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"exp032 cacheが無い: {path}（LightGBM実行可能環境で先に生成する）"
        )
    data = np.load(path)
    y = data["y"]
    return y, ({name: data[f"oof_{name}"] for name in BASE_MEMBERS}), \
        ({name: data[f"te_{name}"] for name in BASE_MEMBERS})


def candidate_oof(seed, cache=True):
    path = os.path.join(OUT_DIR, f"_h47_buying_center_seed{seed}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["y"], d["oof"], d["test"]
    _, _, X, y, Xte = _load_features()
    cols = list(BUYING_CENTER_COLS)
    oof, tests = np.zeros(len(y)), []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(folds.split(X, y), 1):
        print(f"    H47 fold {k}/5 ...", flush=True)
        p, q = _cross_lr(X.iloc[tr][cols], y[tr], X.iloc[va][cols], Xte[cols])
        oof[va], tests = p, tests + [q]
    test_pred = np.mean(tests, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def run(seeds):
    train, test, _, _, _ = _load_features()
    trf, tef = make_buying_center_features(train), make_buying_center_features(test)
    prevalence = pd.DataFrame({
        "train_rate": trf.mean(), "test_rate": tef.mean(),
        "train_count": trf.sum(), "test_count": tef.sum(),
    })
    print("=== H47 特徴分布（ラベル非参照） ===")
    print(prevalence.to_string(float_format=lambda x: f"{x:.3f}"))
    prevalence.to_csv(os.path.join(OUT_DIR, "_h47_feature_stability.csv"))

    rows, weights, correlations, payloads = [], [], [], []
    for position, seed in enumerate(seeds, 1):
        print(f"\n=== seed {seed} ({position}/{len(seeds)}) ===", flush=True)
        y, oof, te = load_base_predictions(seed)
        yy, cand_oof, cand_test = candidate_oof(seed)
        assert np.array_equal(y, yy)
        correlations.append(pd.Series(
            {n: spearmanr(cand_oof, oof[n]).statistic for n in BASE_MEMBERS},
            name=seed,
        ))
        all_oof, all_te = {**oof, NAME: cand_oof}, {**te, NAME: cand_test}
        payloads.append((seed, y, all_oof, all_te))
        base, _, _, _ = blend_oof(y, all_oof, all_te, seed, None,
                                  experts=BASE_MEMBERS)
        candidate, _, w, _ = blend_oof(y, all_oof, all_te, seed, None,
                                       experts=BASE_MEMBERS + (NAME,))
        rows.extend([
            dict(seed=seed, variant="exp032", **_scores(y, base)),
            dict(seed=seed, variant="exp032+H47", **_scores(y, candidate)),
            dict(seed=seed, variant="H47単体", **_scores(y, cand_oof)),
        ])
        weights.append(pd.Series(w, index=BASE_MEMBERS + (NAME,), name=seed))

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT_DIR, "_h47_5seed_scores.csv"), index=False)
    base = scores[scores.variant == "exp032"].sort_values("seed").reset_index(drop=True)
    cand = scores[scores.variant == "exp032+H47"].sort_values("seed").reset_index(drop=True)
    base.attrs["n_feat"], cand.attrs["n_feat"] = 8, 9
    result, deltas = verdict(base, cand)
    print("\n" + format_report(
        "H47", "購買主体の完成度7列LR Expertをexp032へ追加",
        base, cand, result, deltas,
    ))

    corr = pd.DataFrame(correlations)
    corr.to_csv(os.path.join(OUT_DIR, "_h47_5seed_corr.csv"))
    print("\n=== H47と既存ExpertのSpearman（seed平均） ===")
    print(corr.mean().sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))

    W = pd.DataFrame(weights).div(pd.DataFrame(weights).sum(axis=1), axis=0) * 100
    W.to_csv(os.path.join(OUT_DIR, "_h47_5seed_weights.csv"))
    print(f"\nH47メタ重み: {W[NAME].mean():.2f}% ± {W[NAME].std():.2f}%")

    fixed_rows = []
    for alpha in FIXED_ALPHAS:
        for seed, y, oof, te in payloads:
            bp, _, _, _ = blend_oof(y, oof, te, seed, alpha, experts=BASE_MEMBERS)
            cp, _, w, _ = blend_oof(y, oof, te, seed, alpha,
                                    experts=BASE_MEMBERS + (NAME,))
            sb, sc = _scores(y, bp), _scores(y, cp)
            fixed_rows.append(dict(
                alpha=alpha, seed=seed,
                d_auc=sc["auc"] - sb["auc"], d_ap=sc["ap"] - sb["ap"],
                d_f1=sc["f1"] - sb["f1"],
                weight_pct=100 * w[-1] / (w.sum() or 1),
            ))
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(os.path.join(OUT_DIR, "_h47_fixed_alpha_scores.csv"), index=False)
    print("\n=== 固定alpha感度 ===")
    for alpha, d in fixed.groupby("alpha"):
        print(f"alpha={alpha:.3f}: dAP={d.d_ap.mean():+.4f} "
              f"({int((d.d_ap > 0).sum())}/{len(d)}), "
              f"dAUC={d.d_auc.mean():+.4f} "
              f"({int((d.d_auc > 0).sum())}/{len(d)}), "
              f"dF1={d.d_f1.mean():+.4f}, weight={d.weight_pct.mean():.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    run(parser.parse_args().seeds)
