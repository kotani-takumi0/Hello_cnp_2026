"""H77: how often does select_alpha pick a harmful alpha, and what it costs.

``meta_blend`` shrinks the meta weights toward zero and its own docstring warns
that this direction is harmful -- it drives the solution toward the single
strongest expert and kills the orthogonal E2 -- while asserting that
``select_alpha`` "always" lands on the bottom of the grid so the penalty never
actually bites.  The 5-repeat holdout screen of the H70-H76 arms contradicts
that: alpha=1.0 was selected in 2 of 5 repeats, and those two repeats were the
only ones where the blend was badly miscalibrated (logloss 0.35 vs 0.20-0.25).

This script measures the selection directly on the cached expert predictions,
without any holdout, so the failure mode can be counted rather than inferred.

  * per outer fold: the alpha that select_alpha picks, and the inner AP curve
  * per seed: the OOF blend under auto selection, under the 1-SE rule, and
    under each fixed alpha
  * the weight that E2_survey ends up with, since that is the documented
    casualty of over-shrinkage
  * how many of the 800 submitted test labels move when the rule changes

The OOF numbers here are not a substitute for the strict holdout -- alpha is
selected inside each fold's training part, so there is no evaluation leak, but
reading "0.003 is best" off the table and then adopting it would be selection
on the evaluation set.  The 1-SE rule is the version that needs no such choice.

Run:
  .venv/bin/python exp/diagnose_meta_alpha.py --config h2h7both4 \\
      --seeds 0 1 2 3 4 5 6 7 8 9 42
  .venv/bin/python exp/diagnose_meta_alpha.py --config ratio --no-fold-curve
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS  # noqa: E402
from meta_blend import (  # noqa: E402
    ALPHA_GRID, apply_meta, fit_meta, select_alpha, select_alpha_1se, to_logit,
)

# 本命（H2+H7-both4 の8本構成）と、監査 20260829 が使った finance-ratios 構成。
CONFIGS = {
    "h2h7both4": ("exp/_experts_seed{seed}_concatemb_crosslr_lin0.03"
                  "_h2h7both4v2eq.npz"),
    "ratio": "exp/_experts_seed{seed}_ratio_concatemb_crosslr_lin0.03.npz",
}
EXPERTS = ("E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear")


def _load(cache, seed):
    data = np.load(cache.format(seed=seed), allow_pickle=True)
    y = data["y"]
    Z = to_logit(np.column_stack([data[f"oof_{n}"] for n in EXPERTS]))
    Zte = to_logit(np.column_stack([data[f"te_{n}"] for n in EXPERTS]))
    return y, Z, Zte


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _blend(y, Z, seed, alpha=None, rule="auto", Zte=None):
    """blend_oof と同じ外側5foldでOOF合成を作る。

    alpha を渡せば固定。渡さなければ fold ごとに選ぶ（rule="auto" が現行、
    rule="1se" は最良から1標準誤差以内で最小の alpha に留める版）。
    """
    out = np.zeros(len(y))
    picked, weights, test_parts = [], [], []
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        if alpha is not None:
            a = alpha
        elif rule == "1se":
            a = select_alpha_1se(Z[tr], y[tr], seed)
        else:
            a = select_alpha(Z[tr], y[tr], seed=seed)[0]
        w, b = fit_meta(Z[tr], y[tr], alpha=a)
        out[va] = apply_meta(Z[va], w, b)
        picked.append(a)
        weights.append(w)
        if Zte is not None:
            test_parts.append(apply_meta(Zte, w, b))
    weight = np.mean(weights, axis=0)
    share = weight / weight.sum() if weight.sum() > 0 else weight
    test = np.mean(test_parts, axis=0) if test_parts else None
    return out, picked, dict(zip(EXPERTS, share)), test


def _labels_at_oof_best(y, blend, test):
    """OOF最適閾値（本番の既定 oof_best）で test を2値化する。"""
    threshold = THS[int(np.argmax(
        [f1_score(y, (blend >= t).astype(int)) for t in THS]))]
    return (test >= threshold).astype(int), float(threshold)


def _row(seed, label, picked, y, blend, share, labels, threshold):
    return {
        "seed": seed, "alpha": label, "picked": str(picked),
        "auc": roc_auc_score(y, blend),
        "ap": average_precision_score(y, blend),
        "f1": _best_f1(y, blend), "logloss": log_loss(y, blend),
        "E2_share": share["E2_survey"], "test_threshold": threshold,
        "test_positives": int(labels.sum()),
    }


def _fold_curve(y, Z, seed):
    """外側fold ごとに、内側CVのAPを alpha 全点で測る。"""
    rows = []
    for fold, (tr, _) in enumerate(StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y)):
        curve = {}
        for alpha in ALPHA_GRID:
            inner = np.zeros(len(tr))
            for itr, iva in StratifiedKFold(
                    5, shuffle=True, random_state=seed).split(Z[tr], y[tr]):
                w, b = fit_meta(Z[tr][itr], y[tr][itr], alpha=alpha)
                inner[iva] = apply_meta(Z[tr][iva], w, b)
            curve[alpha] = average_precision_score(y[tr], inner)
        best = max(curve, key=curve.get)
        rows.append({
            "seed": seed, "fold": fold, "picked": best,
            "best_ap": curve[best], "ap_at_min": curve[ALPHA_GRID[0]],
            "gap": curve[best] - curve[ALPHA_GRID[0]],
            **{f"ap_{a:g}": v for a, v in curve.items()},
        })
    return rows


def _order():
    return ["auto", "1se"] + [f"{a:g}" for a in ALPHA_GRID]


def _report_folds(fold_frame, n_seeds):
    print(f"=== fold ごとに選ばれた alpha "
          f"({len(fold_frame)} fold = {n_seeds}seed x 5fold) ===")
    for alpha, n in fold_frame["picked"].value_counts().sort_index().items():
        print(f"  alpha={alpha:<6g} {n:2d} fold ({n / len(fold_frame):.0%})")
    big = fold_frame[fold_frame["picked"] >= 0.3]
    print(f"\n  alpha>=0.3 を選んだ fold: {len(big)}/{len(fold_frame)}")
    print(f"  最小alphaに対する内側APの上積み: 平均 "
          f"{fold_frame['gap'].mean():+.5f}  "
          f"最大 {fold_frame['gap'].max():+.5f}")
    print("\n=== 内側AP曲線（fold平均）===")
    curve_cols = [c for c in fold_frame.columns if c.startswith("ap_")]
    print(fold_frame[curve_cols].mean().to_string(
        float_format=lambda v: f"{v:.5f}"))


def _report_seeds(seed_frame, flip_rows):
    order = _order()
    print("\n=== seed ごと ===")
    pivot = seed_frame.pivot(index="alpha", columns="seed",
                             values=["ap", "logloss", "f1", "E2_share"])
    for metric in ("ap", "logloss", "f1", "E2_share"):
        print(f"\n--- {metric} ---")
        print(pivot[metric].reindex(order).to_string(
            float_format=lambda v: f"{v:.4f}"))

    print("\n=== seed 平均 ===")
    print(seed_frame.groupby("alpha")[
        ["auc", "ap", "f1", "logloss", "E2_share"]].mean().reindex(
            order).to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n=== auto を基準にしたペア差 ===")
    base = seed_frame[seed_frame["alpha"] == "auto"].set_index("seed")
    for label in order[1:]:
        other = seed_frame[seed_frame["alpha"] == label].set_index("seed")
        line = [f"  vs alpha={label:<6s}"]
        for metric in ("auc", "ap", "f1"):
            delta = other[metric] - base[metric]
            line.append(f"{metric} {delta.mean():+.5f} "
                        f"({int((delta > 0).sum())}/{len(delta)})")
        print("  ".join(line))

    print("\n=== auto の提出ラベルから何行動くか（test 800行、OOF最適閾値）===")
    print(pd.DataFrame(flip_rows).set_index("seed").to_string())
    print("  1行の正誤は Public F1 を約0.006〜0.007 動かす"
          "（expert_architecture_audit_20260829.md §7c）")


def main(seeds, cache, do_folds=True):
    seed_rows, fold_rows, flip_rows = [], [], []
    for seed in seeds:
        y, Z, Zte = _load(cache, seed)
        test_labels = {}

        for label, rule in (("auto", "auto"), ("1se", "1se")):
            blend, picked, share, test = _blend(
                y, Z, seed, rule=rule, Zte=Zte)
            labels, threshold = _labels_at_oof_best(y, blend, test)
            test_labels[label] = labels
            seed_rows.append(
                _row(seed, label, picked, y, blend, share, labels, threshold))

        for alpha in ALPHA_GRID:
            blend, _, share, test = _blend(y, Z, seed, alpha=alpha, Zte=Zte)
            labels, threshold = _labels_at_oof_best(y, blend, test)
            test_labels[f"{alpha:g}"] = labels
            seed_rows.append(
                _row(seed, f"{alpha:g}", "", y, blend, share, labels,
                     threshold))

        flip_rows.append({
            "seed": seed,
            **{f"vs_{k}": int((test_labels["auto"] != v).sum())
               for k, v in test_labels.items() if k != "auto"},
        })
        if do_folds:
            fold_rows.extend(_fold_curve(y, Z, seed))

    seed_frame = pd.DataFrame(seed_rows)
    if do_folds:
        _report_folds(pd.DataFrame(fold_rows), len(seeds))
        pd.DataFrame(fold_rows).to_csv("exp/meta_alpha_by_fold.csv",
                                       index=False)
    _report_seeds(seed_frame, flip_rows)
    seed_frame.to_csv("exp/meta_alpha_by_seed.csv", index=False)
    print("\nsaved: exp/meta_alpha_by_seed.csv"
          + (", exp/meta_alpha_by_fold.csv" if do_folds else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ratio", choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 42])
    parser.add_argument("--no-fold-curve", action="store_true",
                        help="内側AP曲線の掃引を省いて seed 比較だけ出す")
    args = parser.parse_args()
    main(args.seeds, CONFIGS[args.config], do_folds=not args.no_fold_curve)
