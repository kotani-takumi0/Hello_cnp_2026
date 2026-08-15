"""H39: E7のNeed/Capacity軸へDX実行意欲(Intent)を1列だけ追加する。

現行exp032のE7は、不満集約2軸と財務5軸、その積5本を持つ低次元LR。
EDAで作った3軸のうちNeed/CapacityはすでにE7にあるため、新しい3軸モデルを
9本目として足さず、E7の12列を維持したままLLMの実行スコアだけを13列目へ足す。

比較:
  M0: exp032（現行E7 12列）
  M1: exp032（E7だけを E7+Intent 13列へ置換）

実行:
  python3 exp/compare_e7_intent.py --mode diag --seed 42
  OMP_NUM_THREADS=4 python3 exp/compare_e7_intent.py --mode compare --n-seeds 10

compareではauto-alphaに加え固定alphaも確認する。固定alphaで残らない改善は、
候補の追加ではなくalpha選択境界が動いただけの可能性があるため15repへ進めない。
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

from cross_features import ALL_CROSS_COLS  # noqa: E402
from decision import format_report, verdict  # noqa: E402
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, OUT_DIR, _cross_lr, _scores, blend_oof,
    build_features, compute_expert_preds,
)
from llm_features import (  # noqa: E402
    LLM_COLS, TEST_LLM, TRAIN_LLM, add_llm_axes,
)

INTENT_COL = "llm_dx_execution_score"
E7_INTENT_COLS = tuple(ALL_CROSS_COLS) + (INTENT_COL,)
BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
FIXED_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03)
CORR_GATE = 0.90


def _load_features():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    X = add_llm_axes(train, X, TRAIN_LLM)
    Xte = add_llm_axes(test, Xte, TEST_LLM)
    return train, test, X, y, Xte[X.columns]


def e7_intent_oof(seed, cache=True):
    """現行E7と同じ5fold・同じLRで、Intent追加版のOOF/test予測を作る。"""
    path = os.path.join(OUT_DIR, f"_h39_e7_intent_seed{seed}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["y"], d["oof"], d["test"]

    _, _, X, y, Xte = _load_features()
    cols = list(E7_INTENT_COLS)
    oof, tests = np.zeros(len(y)), []
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    H39 fold {k}/5 ...", flush=True)
        p, q = _cross_lr(
            X.iloc[tr][cols], y[tr], X.iloc[va][cols], Xte[cols]
        )
        oof[va] = p
        tests.append(q)
    test_pred = np.mean(tests, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def _load_base(seed):
    path = os.path.join(
        OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr_lin0.03.npz"
    )
    if not os.path.exists(path):
        compute_expert_preds(
            seed, concat_embed=True, cross=True, e7_model="lr",
            linear=True, linear_c=0.03,
        )
    d = np.load(path)
    y = d["y"]
    oof = {n: d[f"oof_{n}"] for n in BASE_MEMBERS}
    test = {n: d[f"te_{n}"] for n in BASE_MEMBERS}
    return y, oof, test


def run_diag(seed):
    train, test, _, _, _ = _load_features()
    y, base_oof, _ = _load_base(seed)
    yy, candidate_oof, _ = e7_intent_oof(seed, cache=False)
    assert np.array_equal(y, yy)

    stability = pd.DataFrame({
        "train": train[["企業ID"]].merge(
            pd.read_csv(TRAIN_LLM, usecols=["企業ID", INTENT_COL]),
            on="企業ID", how="left",
        )[INTENT_COL].describe(),
        "test": test[["企業ID"]].merge(
            pd.read_csv(TEST_LLM, usecols=["企業ID", INTENT_COL]),
            on="企業ID", how="left",
        )[INTENT_COL].describe(),
    })
    print("\n=== Intent train/test安定性 ===")
    print(stability.to_string(float_format=lambda v: f"{v:.4f}"))

    rows = {
        "E7_current": _scores(y, base_oof[E7_NAME]),
        "E7_plus_intent": _scores(y, candidate_oof),
    }
    table = pd.DataFrame(rows).T
    table["AP倍"] = table["ap"] / y.mean()
    print(f"\n=== H39 E7置換単体診断 (seed={seed}) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    table.to_csv(os.path.join(OUT_DIR, "_h39_diag_scores.csv"))

    corr = pd.Series({
        n: spearmanr(candidate_oof, p).statistic for n, p in base_oof.items()
    }, name="E7_plus_intent")
    corr.loc["最大|r|"] = corr.abs().max()
    print(f"\n=== E7+Intentと既存exp032 8本のSpearman相関 (gate={CORR_GATE}) ===")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))
    corr.to_frame().T.to_csv(os.path.join(OUT_DIR, "_h39_diag_corr.csv"))
    print(f"\n現行E7との相関: {corr[E7_NAME]:.3f}")
    print("置換候補なので相関0.90超でも即REJECTにはせず、同一seedペア差を確認する。")


def run_compare(n_seeds):
    rows, weights, payloads = [], [], []
    for seed in range(n_seeds):
        print(f"\n=== seed {seed}/{n_seeds - 1} ===", flush=True)
        y, base_oof, base_test = _load_base(seed)
        yy, candidate_oof, candidate_test = e7_intent_oof(seed)
        assert np.array_equal(y, yy)

        candidate_dict = {**base_oof, E7_NAME: candidate_oof}
        candidate_test_dict = {**base_test, E7_NAME: candidate_test}
        payloads.append(
            (seed, y, base_oof, base_test, candidate_dict, candidate_test_dict)
        )

        bp, _, _, _ = blend_oof(
            y, base_oof, base_test, seed, None, experts=BASE_MEMBERS
        )
        cp, _, w, _ = blend_oof(
            y, candidate_dict, candidate_test_dict, seed, None,
            experts=BASE_MEMBERS,
        )
        rows.append(dict(seed=seed, variant="exp032", **_scores(y, bp)))
        rows.append(dict(seed=seed, variant="exp032_E7+Intent", **_scores(y, cp)))
        rows.append(dict(seed=seed, variant="E7_current", **_scores(y, base_oof[E7_NAME])))
        rows.append(dict(seed=seed, variant="E7+Intent", **_scores(y, candidate_oof)))
        weights.append(pd.Series(w, index=BASE_MEMBERS, name=seed))

    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(OUT_DIR, "_h39_10seed_scores.csv"), index=False)
    base = scores[scores.variant == "exp032"].sort_values("seed").reset_index(drop=True)
    cand = scores[
        scores.variant == "exp032_E7+Intent"
    ].sort_values("seed").reset_index(drop=True)
    base.attrs["n_feat"] = cand.attrs["n_feat"] = len(BASE_MEMBERS)
    result, deltas = verdict(base, cand)
    print("\n" + format_report(
        "H39", "exp032のE7へIntent 1列を追加（8本構成のまま置換）",
        base, cand, result, deltas,
    ))

    old = scores[scores.variant == "E7_current"].sort_values("seed")
    new = scores[scores.variant == "E7+Intent"].sort_values("seed")
    old.attrs["n_feat"], new.attrs["n_feat"] = 12, 13
    e7_result, e7_deltas = verdict(old.reset_index(drop=True), new.reset_index(drop=True))
    print("\n" + format_report(
        "H39-E7", "E7単体へIntent 1列を追加", old, new,
        e7_result, e7_deltas,
    ))

    W = pd.DataFrame(weights)
    Wn = W.div(W.sum(axis=1), axis=0) * 100
    Wn.to_csv(os.path.join(OUT_DIR, "_h39_10seed_weights.csv"))
    print("\n=== E7+Intent版メタ重み (正規化%, 10seed) ===")
    for c in Wn.mean().sort_values(ascending=False).index:
        print(f"  {c:16s} {Wn[c].mean():5.1f}% ±{Wn[c].std():4.1f}")

    fixed_rows = []
    print("\n=== 固定alpha感度（置換効果がalpha選択に依存しないか） ===")
    for alpha in FIXED_ALPHA_GRID:
        for seed, y, bo, bt, co, ct in payloads:
            bp, _, _, _ = blend_oof(y, bo, bt, seed, alpha, experts=BASE_MEMBERS)
            cp, _, w, _ = blend_oof(y, co, ct, seed, alpha, experts=BASE_MEMBERS)
            sb, sc = _scores(y, bp), _scores(y, cp)
            fixed_rows.append(dict(
                alpha=alpha, seed=seed,
                d_auc=sc["auc"] - sb["auc"],
                d_ap=sc["ap"] - sb["ap"],
                d_f1=sc["f1"] - sb["f1"],
                e7_weight_pct=100 * w[BASE_MEMBERS.index(E7_NAME)] / (w.sum() or 1),
            ))
        d = pd.DataFrame(fixed_rows)
        d = d[d.alpha == alpha]
        print(f"  alpha={alpha:.3f}: ΔAP {d.d_ap.mean():+.4f} "
              f"({int((d.d_ap > 0).sum())}/{len(d)}) / "
              f"ΔAUC {d.d_auc.mean():+.4f} "
              f"({int((d.d_auc > 0).sum())}/{len(d)}) / "
              f"ΔF1 {d.d_f1.mean():+.4f} "
              f"({int((d.d_f1 > 0).sum())}/{len(d)}) / "
              f"E7重み {d.e7_weight_pct.mean():.2f}%")
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(os.path.join(OUT_DIR, "_h39_fixed_alpha_scores.csv"), index=False)

    fixed_pass = []
    for alpha, d in fixed.groupby("alpha"):
        if ((d.d_ap > 0).sum() >= np.ceil(.75 * len(d))
                and (d.d_auc > 0).sum() >= np.ceil(.70 * len(d))
                and d.d_ap.mean() > 0 and d.d_f1.mean() >= -0.002):
            fixed_pass.append(float(alpha))
    proceed = result.startswith("ACCEPT") and bool(fixed_pass)
    print(f"\n15rep前ゲート: {'通過' if proceed else 'STOP'}")
    print(f"  auto-alpha={result}")
    print(f"  固定alpha通過={fixed_pass or 'なし'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diag", "compare"), default="diag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "diag":
        run_diag(args.seed)
    else:
        run_compare(args.n_seeds)
