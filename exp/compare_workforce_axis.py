"""H35 研修対象人材・知識労働集約度エキスパートの診断と10seed比較。

実行:
  python3 exp/compare_workforce_axis.py --mode diag --seed 42
  OMP_NUM_THREADS=4 python3 exp/compare_workforce_axis.py --mode compare --n-seeds 10

diagでLR/LGBMを比較するが、4軸自体は結果を見る前に
``workforce_axis_features.py``で固定している。compareではexp032にE11を追加し、
auto-alphaと固定alphaの両方で残差寄与を判定する。
"""
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

from decision import format_report, verdict  # noqa: E402
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, OUT_DIR, _cross_lr, _lgbm, _scores,
    blend_oof, build_features, compute_expert_preds,
)
from workforce_axis_features import (  # noqa: E402
    WORKFORCE_AXIS_COLS, WORKFORCE_INTERACTION_COLS,
    add_workforce_axes, add_workforce_interaction,
    make_workforce_axes, make_workforce_interaction,
)

BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
CORR_GATE = 0.90
FIXED_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03)
CANDIDATES = {
    "axes": dict(name="E11_workforce", cols=WORKFORCE_AXIS_COLS,
                 prefix="h35", label="H35", desc="研修対象人材・知識労働集約度4軸"),
    "interaction": dict(name="E11_workforce_x", cols=WORKFORCE_INTERACTION_COLS,
                         prefix="h35b", label="H35b",
                         desc="研修対象人口の親2本＋積"),
}


def _load_features(candidate="axes"):
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    adder = add_workforce_axes if candidate == "axes" else add_workforce_interaction
    X = adder(train, X)
    Xte = adder(test, Xte)
    return train, test, X, y, Xte[X.columns]


def _fit(Xtr, ytr, Xva, yva, Xte, seed, model):
    if model == "lr":
        return _cross_lr(Xtr, ytr, Xva, Xte)
    return _lgbm(Xtr, ytr, Xva, yva, Xte, seed)


def e11_oof(seed, model, candidate="axes", cache=True):
    cfg = CANDIDATES[candidate]
    path = os.path.join(OUT_DIR, f"_{cfg['prefix']}_e11_seed{seed}_{model}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["y"], d["oof"], d["test"]

    _, _, X, y, Xte = _load_features(candidate)
    cols = list(cfg["cols"])
    oof, tests = np.zeros(len(y)), []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    {cfg['label']} {model} fold {k}/5 ...", flush=True)
        p, q = _fit(X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va],
                    Xte[cols], seed, model)
        oof[va] = p
        tests.append(q)
    test_pred = np.mean(tests, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def _stability_table(train, test, candidate):
    maker = make_workforce_axes if candidate == "axes" else make_workforce_interaction
    a, b = maker(train), maker(test)
    rows = []
    for col in CANDIDATES[candidate]["cols"]:
        rows.append(dict(
            feature=col,
            train_missing=float(a[col].isna().mean()),
            test_missing=float(b[col].isna().mean()),
            train_mean=float(a[col].mean()),
            test_mean=float(b[col].mean()),
            train_p10=float(a[col].quantile(.1)),
            train_p50=float(a[col].quantile(.5)),
            train_p90=float(a[col].quantile(.9)),
            test_p10=float(b[col].quantile(.1)),
            test_p50=float(b[col].quantile(.5)),
            test_p90=float(b[col].quantile(.9)),
        ))
    return pd.DataFrame(rows).set_index("feature")


def run_diag(seed, candidate):
    cfg = CANDIDATES[candidate]
    e11_name, prefix = cfg["name"], cfg["prefix"]
    base_path = os.path.join(
        OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr_lin0.03.npz"
    )
    if not os.path.exists(base_path):
        compute_expert_preds(seed, concat_embed=True, cross=True,
                             e7_model="lr", linear=True, linear_c=0.03)
    base_npz = np.load(base_path)
    y = base_npz["y"]
    base = {n: base_npz[f"oof_{n}"] for n in BASE_MEMBERS}

    train, test, _, _, _ = _load_features(candidate)
    stability = _stability_table(train, test, candidate)
    print(f"\n=== {cfg['label']} train/test安定性 ===")
    print(stability.to_string(float_format=lambda v: f"{v:.4f}"))
    stability.to_csv(os.path.join(OUT_DIR, f"_{prefix}_diag_stability.csv"))

    rows = {n: _scores(y, p) for n, p in base.items()}
    preds = {}
    for model in ("lr", "lgbm"):
        print(f"\nE11({model})を学習", flush=True)
        yy, oof, _ = e11_oof(seed, model, candidate, cache=False)
        assert np.array_equal(y, yy)
        preds[model] = oof
        rows[f"{e11_name}({model})"] = _scores(y, oof)

    table = pd.DataFrame(rows).T
    table["AP倍"] = table["ap"] / y.mean()
    print(f"\n=== {cfg['label']}単体診断 (seed={seed}) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    table.to_csv(os.path.join(OUT_DIR, f"_{prefix}_diag_scores.csv"))

    pick = max(("lr", "lgbm"),
               key=lambda m: table.loc[f"{e11_name}({m})", "ap"])
    corr = pd.DataFrame({
        model: {n: spearmanr(pred, base[n]).statistic for n in BASE_MEMBERS}
        for model, pred in preds.items()
    }).T
    corr["最大|r|"] = corr.abs().max(axis=1)
    print(f"\n採用モデル候補: {pick}")
    print(f"\n=== 既存exp032 8本とのSpearman相関 (gate={CORR_GATE}) ===")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))
    corr.to_csv(os.path.join(OUT_DIR, f"_{prefix}_diag_corr.csv"))
    mx = corr.loc[pick, "最大|r|"]
    print(f"\n相関ゲート: {mx:.3f} -> " + ("REJECT" if mx >= CORR_GATE else "通過"))
    print(f"単体AP: {table.loc[f'{e11_name}({pick})', 'ap']:.4f} "
          f"({table.loc[f'{e11_name}({pick})', 'AP倍']:.2f}倍)")
    print(f"次の比較コマンド: python3 exp/compare_workforce_axis.py "
          f"--mode compare --n-seeds 10 --model {pick} --candidate {candidate}")


def run_compare(n_seeds, model, candidate):
    cfg = CANDIDATES[candidate]
    e11_name, prefix = cfg["name"], cfg["prefix"]
    rows, weights, payloads = [], [], []
    for seed in range(n_seeds):
        print(f"\n=== seed {seed}/{n_seeds - 1} ===", flush=True)
        y, oof, te = compute_expert_preds(
            seed, concat_embed=True, cross=True, e7_model="lr",
            linear=True, linear_c=0.03,
        )
        yy, candidate_oof, candidate_test = e11_oof(seed, model, candidate)
        assert np.array_equal(y, yy)
        oof = {**oof, e11_name: candidate_oof}
        te = {**te, e11_name: candidate_test}
        payloads.append((seed, y, oof, te))

        base_pred, _, _, _ = blend_oof(
            y, oof, te, seed, None, experts=BASE_MEMBERS
        )
        candidate_members = BASE_MEMBERS + (e11_name,)
        cand_pred, _, w, _ = blend_oof(
            y, oof, te, seed, None, experts=candidate_members
        )
        rows.append(dict(seed=seed, variant="exp032", **_scores(y, base_pred)))
        rows.append(dict(seed=seed, variant=f"exp032+{cfg['label']}", **_scores(y, cand_pred)))
        rows.append(dict(seed=seed, variant=f"{cfg['label']}単体", **_scores(y, candidate_oof)))
        weights.append(pd.Series(w, index=candidate_members, name=seed))

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT_DIR, f"_{prefix}_10seed_scores.csv"), index=False)
    base = scores[scores.variant == "exp032"].sort_values("seed").reset_index(drop=True)
    cand = scores[scores.variant == f"exp032+{cfg['label']}"] .sort_values("seed").reset_index(drop=True)
    base.attrs["n_feat"], cand.attrs["n_feat"] = len(BASE_MEMBERS), len(BASE_MEMBERS) + 1
    result, deltas = verdict(base, cand)
    print("\n" + format_report(
        cfg["label"], f"exp032へ{cfg['desc']}エキスパートを追加",
        base, cand, result, deltas,
    ))

    W = pd.DataFrame(weights)
    Wn = W.div(W.sum(axis=1), axis=0) * 100
    Wn.to_csv(os.path.join(OUT_DIR, f"_{prefix}_10seed_weights.csv"))
    print(f"\n=== exp032+{cfg['label']} メタ重み (正規化%, 10seed) ===")
    for col in Wn.mean().sort_values(ascending=False).index:
        print(f"  {col:16s} {Wn[col].mean():5.1f}% ±{Wn[col].std():4.1f}")

    fixed_rows = []
    print("\n=== 固定alpha感度（E11自身の残差寄与） ===")
    for alpha in FIXED_ALPHA_GRID:
        for seed, y, oof, te in payloads:
            bp, _, _, _ = blend_oof(y, oof, te, seed, alpha,
                                     experts=BASE_MEMBERS)
            cp, _, w, _ = blend_oof(y, oof, te, seed, alpha,
                                     experts=BASE_MEMBERS + (e11_name,))
            sb, sc = _scores(y, bp), _scores(y, cp)
            fixed_rows.append(dict(
                alpha=alpha, seed=seed,
                d_auc=sc["auc"] - sb["auc"],
                d_ap=sc["ap"] - sb["ap"],
                d_f1=sc["f1"] - sb["f1"],
                e11_weight_pct=100 * w[-1] / (w.sum() or 1),
            ))
        d = pd.DataFrame(fixed_rows)
        d = d[d.alpha == alpha]
        print(f"  alpha={alpha:.3f}: ΔAP {d.d_ap.mean():+.4f} "
              f"({int((d.d_ap > 0).sum())}/{len(d)}) / "
              f"ΔAUC {d.d_auc.mean():+.4f} "
              f"({int((d.d_auc > 0).sum())}/{len(d)}) / "
              f"ΔF1 {d.d_f1.mean():+.4f} "
              f"({int((d.d_f1 > 0).sum())}/{len(d)}) / "
              f"E11重み {d.e11_weight_pct.mean():.2f}%")
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(os.path.join(OUT_DIR, f"_{prefix}_fixed_alpha_scores.csv"), index=False)

    # 15rep前ゲート: auto-alphaの正式基準に加え、少なくとも1つの固定alphaで
    # AP正率>=8/10・平均AP>0・平均F1>=-0.002を要求する。
    grouped = fixed.groupby("alpha")
    fixed_pass = []
    for alpha, d in grouped:
        if ((d.d_ap > 0).sum() >= 8 and (d.d_auc > 0).sum() >= 7
                and d.d_ap.mean() > 0
                and d.d_f1.mean() >= -0.002):
            fixed_pass.append(float(alpha))
    auto_ok = result.startswith("ACCEPT")
    proceed = auto_ok and bool(fixed_pass)
    print(f"\n15rep前ゲート: {'通過' if proceed else 'STOP'}")
    print(f"  auto-alpha={result}")
    print(f"  固定alpha通過={fixed_pass or 'なし'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diag", "compare"), default="diag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--model", choices=("lgbm", "lr"), default="lr")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="axes")
    args = parser.parse_args()
    if args.mode == "diag":
        run_diag(args.seed, args.candidate)
    else:
        run_compare(args.n_seeds, args.model, args.candidate)
