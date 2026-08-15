"""H34 収益力×DX意欲エキスパートの診断と10seed合成比較。

段階1 (diag):
  - 14列の LR / LightGBM を同一5foldで比較
  - 単体APと既存exp032の8本とのSpearman相関を確認

段階2 (compare):
  - exp032（concat embedding + E7 LR + E0b LR）を基準にする
  - H34を9本目として足し、同一seedの外側メタOOFで比較
  - ここでは完全ホールドアウト15repは実行しない

実行:
  python exp/compare_profit_dx_cross.py --mode diag --seed 42
  OMP_NUM_THREADS=4 python exp/compare_profit_dx_cross.py --mode compare --n-seeds 10
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decision import format_report, verdict  # noqa: E402
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, LINEAR_C, OUT_DIR, THS, _cross_lr, _lgbm,
    _scores, blend_oof, build_features, compute_expert_preds,
)
from profit_dx_cross_features import (  # noqa: E402
    ALL_PROFIT_DX_COLS, add_profit_dx_cross_features,
)

E10_NAME = "E10_profit_dx"
BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
CORR_GATE = 0.90
FIXED_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03)


def _load_features():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    X = add_profit_dx_cross_features(train, X)
    Xte = add_profit_dx_cross_features(test, Xte)
    return X, y, Xte[X.columns]


def _fit(Xtr, ytr, Xva, yva, Xte, seed, model):
    if model == "lr":
        return _cross_lr(Xtr, ytr, Xva, Xte)
    return _lgbm(Xtr, ytr, Xva, yva, Xte, seed)


def e10_oof(seed, model, cache=True):
    path = os.path.join(OUT_DIR, f"_h34_e10_seed{seed}_{model}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["y"], d["oof"], d["test"]

    X, y, Xte = _load_features()
    cols = list(ALL_PROFIT_DX_COLS)
    oof, tests = np.zeros(len(y)), []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    H34 {model} fold {k}/5 ...", flush=True)
        p, q = _fit(X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va],
                    Xte[cols], seed, model)
        oof[va] = p
        tests.append(q)
    test_pred = np.mean(tests, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def _metric(y, p):
    return _scores(y, p)


def run_diag(seed):
    base_path = os.path.join(
        OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr_lin0.03.npz"
    )
    if not os.path.exists(base_path):
        compute_expert_preds(seed, concat_embed=True, cross=True,
                             e7_model="lr", linear=True, linear_c=0.03)
    base_npz = np.load(base_path)
    y = base_npz["y"]
    base = {n: base_npz[f"oof_{n}"] for n in BASE_MEMBERS}

    rows = {n: _metric(y, p) for n, p in base.items()}
    preds = {}
    for model in ("lgbm", "lr"):
        print(f"\nE10({model})を学習", flush=True)
        yy, oof, _ = e10_oof(seed, model, cache=False)
        assert np.array_equal(y, yy)
        preds[model] = oof
        rows[f"{E10_NAME}({model})"] = _metric(y, oof)

    table = pd.DataFrame(rows).T
    table["AP倍"] = table["ap"] / y.mean()
    print(f"\n=== H34単体診断 (seed={seed}) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    table.to_csv(os.path.join(OUT_DIR, "_h34_diag_scores.csv"))

    pick = max(("lgbm", "lr"), key=lambda m: table.loc[f"{E10_NAME}({m})", "ap"])
    print(f"\n採用モデル候補: {pick} "
          f"(LR-LGBM ΔAP={table.loc[f'{E10_NAME}(lr)', 'ap'] - table.loc[f'{E10_NAME}(lgbm)', 'ap']:+.4f})")

    corr = pd.DataFrame({
        model: {n: spearmanr(pred, base[n]).statistic for n in BASE_MEMBERS}
        for model, pred in preds.items()
    }).T
    corr["最大|r|"] = corr.abs().max(axis=1)
    print(f"\n=== 既存exp032 8本とのSpearman相関 (gate={CORR_GATE}) ===")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))
    corr.to_csv(os.path.join(OUT_DIR, "_h34_diag_corr.csv"))
    mx = corr.loc[pick, "最大|r|"]
    print(f"\n相関ゲート: {mx:.3f} -> "
          + ("REJECT" if mx >= CORR_GATE else "通過"))
    print(f"単体AP: {table.loc[f'{E10_NAME}({pick})', 'ap']:.4f} "
          f"({table.loc[f'{E10_NAME}({pick})', 'AP倍']:.2f}倍)")


def run_compare(n_seeds, model):
    rows, weights, payloads = [], [], []
    for seed in range(n_seeds):
        print(f"\n=== seed {seed}/{n_seeds - 1} ===", flush=True)
        y, oof, te = compute_expert_preds(
            seed, concat_embed=True, cross=True, e7_model="lr",
            linear=True, linear_c=0.03,
        )
        yy, e10_oof_pred, e10_test_pred = e10_oof(seed, model)
        assert np.array_equal(y, yy)
        oof = {**oof, E10_NAME: e10_oof_pred}
        te = {**te, E10_NAME: e10_test_pred}
        payloads.append((seed, y, oof, te))

        base_pred, _, _, _ = blend_oof(
            y, oof, te, seed, None, experts=BASE_MEMBERS
        )
        cand_members = BASE_MEMBERS + (E10_NAME,)
        cand_pred, _, w, _ = blend_oof(
            y, oof, te, seed, None, experts=cand_members
        )
        rows.append(dict(seed=seed, variant="exp032", **_metric(y, base_pred)))
        rows.append(dict(seed=seed, variant="exp032+H34", **_metric(y, cand_pred)))
        rows.append(dict(seed=seed, variant="H34単体", **_metric(y, e10_oof_pred)))
        weights.append(pd.Series(w, index=cand_members, name=seed))

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT_DIR, "_h34_10seed_scores.csv"), index=False)
    base = scores[scores.variant == "exp032"].sort_values("seed").reset_index(drop=True)
    cand = scores[scores.variant == "exp032+H34"].sort_values("seed").reset_index(drop=True)
    base.attrs["n_feat"], cand.attrs["n_feat"] = len(BASE_MEMBERS), len(BASE_MEMBERS) + 1
    result, deltas = verdict(base, cand)
    print("\n" + format_report(
        "H34", "exp032へ収益力×DX意欲エキスパートを追加",
        base, cand, result, deltas,
    ))

    W = pd.DataFrame(weights)
    Wn = W.div(W.sum(axis=1), axis=0) * 100
    Wn.to_csv(os.path.join(OUT_DIR, "_h34_10seed_weights.csv"))
    print("\n=== exp032+H34 メタ重み (正規化%, 10seed) ===")
    for c in Wn.mean().sort_values(ascending=False).index:
        print(f"  {c:16s} {Wn[c].mean():5.1f}% ±{Wn[c].std():4.1f}")
    # 候補追加によって内側CVのalpha選択だけが変わると、候補自身の重みがほぼ0でも
    # auto-alpha OOFが改善し得る。固定alphaでも差が残るかを必ず確認する。
    fixed_rows = []
    print("\n=== 固定alpha感度（候補そのものの残差寄与を確認） ===")
    for alpha in FIXED_ALPHA_GRID:
        for seed, y, oof, te in payloads:
            bp, _, _, _ = blend_oof(y, oof, te, seed, alpha,
                                     experts=BASE_MEMBERS)
            cp, _, w, _ = blend_oof(y, oof, te, seed, alpha,
                                     experts=BASE_MEMBERS + (E10_NAME,))
            sb, sc = _metric(y, bp), _metric(y, cp)
            fixed_rows.append(dict(
                alpha=alpha, seed=seed,
                d_auc=sc["auc"] - sb["auc"],
                d_ap=sc["ap"] - sb["ap"],
                d_f1=sc["f1"] - sb["f1"],
                e10_weight_pct=100 * w[-1] / (w.sum() or 1),
            ))
        f = pd.DataFrame(fixed_rows)
        d = f[f.alpha == alpha]
        print(f"  alpha={alpha:.3f}: ΔAP {d.d_ap.mean():+.4f} "
              f"({int((d.d_ap > 0).sum())}/{len(d)}) / "
              f"ΔAUC {d.d_auc.mean():+.4f} "
              f"({int((d.d_auc > 0).sum())}/{len(d)}) / "
              f"ΔF1 {d.d_f1.mean():+.4f} "
              f"({int((d.d_f1 > 0).sum())}/{len(d)}) / "
              f"E10重み {d.e10_weight_pct.mean():.2f}%")
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(os.path.join(OUT_DIR, "_h34_fixed_alpha_scores.csv"), index=False)
    fixed_ok = bool((fixed.groupby("alpha").d_ap.mean() > 0).any())
    if "弱い" in result and not fixed_ok:
        final = "STOP（auto-alphaだけの弱い改善。固定alphaで再現せず）"
    else:
        final = result
    print(f"\n15repへ進む前判定: {final}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("diag", "compare"), default="diag")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--model", choices=("lgbm", "lr"), default="lr")
    a = p.parse_args()
    if a.mode == "diag":
        run_diag(a.seed)
    else:
        run_compare(a.n_seeds, a.model)
