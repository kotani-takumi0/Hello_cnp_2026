"""H91: does "has short-term borrowings" split the ambiguous band?

The flag is presence, not amount: ``短期借入金 > 0``.  Across all 742 rows it
separates buyers well -- 0.345 without borrowings vs 0.194 with -- so the
question is whether any of that separation survives *inside* the band, which
is where F1 is actually decided.

Two protocols, kept separate:

``screen``
    Descriptive only.  Purchase rate and band AUC by group over seven band
    widths and eleven blend seeds, plus the group-wise calibration check that
    explains how a globally strong flag can be worth nothing inside the band.

``strict``
    Decisive.  Production outer folds (the blend's own 5-fold), never an A/B
    holdout: the band is defined by the blend's own output, and A/B flatters
    exactly that shape (meta_native_and_alpha_selection_20260829.md 4.4).
    Each arm learns its shift ``delta`` -- and, for ``gate_auto``, its band
    edges too -- from the fold's training rows only, on a grid containing 0.
    An arm can therefore decline the gate, and a delta that stays 0 is itself
    the answer.

Arms:

    meta_only        current blend, untouched
    gate_wide        logit += delta on band rows with borrowings (0.1-0.9)
    gate_core        same, on the core band (0.3-0.7) where the screen showed
                     the sign flipping toward "borrowings -> buys"
    gate_auto        band edges also chosen inside the fold
    meta_plus_flag   the flag handed to the meta learner as a 9th input

Run:
  .venv/bin/python exp/compare_short_term_debt_gate.py --mode screen
  .venv/bin/python exp/compare_short_term_debt_gate.py --mode strict \
      --seeds 0 1 2 3 4 5 6 7 8 9 42
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_meta_native import _oof_logit  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from meta_blend import EPS, fit_meta, select_alpha, select_alpha_1se  # noqa: E402

BASELINE = "meta_only"
GATES = ("gate_wide", "gate_core", "gate_auto", "meta_plus_flag")
WIDE = (0.1, 0.9)
CORE = (0.3, 0.7)
# gate_auto が fold 内で選べる帯。screen の走査と同じ刻み。
EDGE_GRID = ((0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8),
             (0.25, 0.75), (0.3, 0.7), (0.35, 0.65))
# ロジット単位のずらし幅。0 を含むので「使わない」も選べる。符号も自由。
DELTA_GRID = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
SCREEN_BANDS = EDGE_GRID


def has_short_term_debt(path="data/train.csv"):
    """短期借入金の**有無**。金額ではない。"""
    return (pd.read_csv(path)["短期借入金"].astype(float).values > 0).astype(float)


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _alpha(Z, y, seed, rule):
    if rule == "1se":
        return select_alpha_1se(Z, y, seed)
    return select_alpha(Z, y, seed=seed)[0]


def _select_gate(train_logit, y_tr, flag_tr, edges_grid):
    """fold の学習行の中だけで (帯, delta) を選ぶ。基準は学習行全体の AP。

    帯を跨いで比べるので、帯内AP ではなく全学習行の AP を使う。ゲートは帯の
    行しか動かさないから、両者は同じ行集合の上で比較できる。
    """
    p_tr = expit(train_logit)
    best_delta, best_edges, best_ap = 0.0, edges_grid[0], -np.inf
    for lo, hi in edges_grid:
        move = flag_tr * ((p_tr > lo) & (p_tr < hi))
        if move.sum() < 10:
            continue
        for delta in DELTA_GRID:
            ap = average_precision_score(y_tr, train_logit + delta * move)
            if ap > best_ap:
                best_delta, best_edges, best_ap = delta, (lo, hi), ap
    return best_delta, best_edges


def _one_seed(y, Z, flag, seed, rule):
    out = {name: np.zeros(len(y)) for name in (BASELINE, *GATES)}
    detail = []
    Zf = np.column_stack([Z, flag, 1.0 - flag])  # 非負重みでも符号が両方向に出る

    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        alpha = _alpha(Z[tr], y[tr], seed, rule)
        weight, intercept = fit_meta(Z[tr], y[tr], alpha=alpha)
        base_va = Z[va] @ weight + intercept
        out[BASELINE][va] = expit(base_va)

        inner = _oof_logit(Z[tr], y[tr], alpha, seed)
        p_va = out[BASELINE][va]
        row = {"seed": seed}

        for name, grid in (("gate_wide", (WIDE,)), ("gate_core", (CORE,)),
                           ("gate_auto", EDGE_GRID)):
            delta, (lo, hi) = _select_gate(inner, y[tr], flag[tr], grid)
            band_va = (p_va > lo) & (p_va < hi)
            out[name][va] = expit(base_va + delta * flag[va] * band_va)
            row[f"delta[{name}]"] = delta
            row[f"帯[{name}]"] = f"{lo}-{hi}"
            row[f"帯内で動いた行[{name}]"] = int((flag[va] * band_va).sum())

        alpha_f = _alpha(Zf[tr], y[tr], seed, rule)
        w_f, b_f = fit_meta(Zf[tr], y[tr], alpha=alpha_f)
        out["meta_plus_flag"][va] = expit(Zf[va] @ w_f + b_f)
        row["重み[有り]"] = float(w_f[-2])
        row["重み[無し]"] = float(w_f[-1])
        detail.append(row)
    return out, detail


def run_strict(seeds, config, rule):
    flag = has_short_term_debt()
    rows, details = [], []
    for seed in seeds:
        y, Z, _ = _load(CONFIGS[config], seed)
        out, detail = _one_seed(y, Z, flag, seed, rule)
        details.extend(detail)
        for name, p in out.items():
            rows.append({"seed": seed, "arm": name,
                         "auc": roc_auc_score(y, p),
                         "ap": average_precision_score(y, p),
                         "f1": _best_f1(y, p)})
    result = pd.DataFrame(rows)
    frame = pd.DataFrame(details)

    print(f"=== fold の中で選ばれた delta（{len(frame)} fold）===")
    print("delta=0 は「このゲートを使わない」という fold 内の判断。\n")
    for name in ("gate_wide", "gate_core", "gate_auto"):
        counts = frame[f"delta[{name}]"].value_counts().sort_index()
        zero = int((frame[f"delta[{name}]"] == 0).sum())
        print(f"  {name:15s} 0を選んだfold {zero:>2}/{len(frame)}  "
              + "  ".join(f"δ={k:+g}: {v}" for k, v in counts.items()))
    print("\n  gate_auto が選んだ帯: "
          + "  ".join(f"{k}: {v}fold"
                      for k, v in frame["帯[gate_auto]"].value_counts().items()))
    print(f"\n  meta_plus_flag の重み  有り {frame['重み[有り]'].mean():.4f}  "
          f"無し {frame['重み[無し]'].mean():.4f}  "
          f"（8本のExpert重みと同じ尺度。0なら使われていない）")

    print("\n=== 本番と同じ外側5fold での成績 ===")
    print(result.groupby("arm")[["auc", "ap", "f1"]].mean()
          .reindex([BASELINE, *GATES]).to_string(float_format=lambda v: f"{v:.5f}"))
    base = result[result.arm == BASELINE].sort_values("seed").reset_index(drop=True)
    print()
    for name in GATES:
        cand = result[result.arm == name].sort_values("seed").reset_index(drop=True)
        line = [f"  {name:15s}"]
        for metric in ("auc", "ap", "f1"):
            d = cand[metric].values - base[metric].values
            line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
        decision, _ = verdict(base, cand)
        print("  ".join(line) + f"   {decision}")
    result.to_csv("exp/short_term_debt_gate.csv", index=False)
    frame.to_csv("exp/short_term_debt_gate_folds.csv", index=False)
    print("\nsaved: exp/short_term_debt_gate.csv, exp/short_term_debt_gate_folds.csv")


def _blend_oof_p(y, Z, seed, rule):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        alpha = _alpha(Z[tr], y[tr], seed, rule)
        w, b = fit_meta(Z[tr], y[tr], alpha=alpha)
        p[va] = expit(Z[va] @ w + b)
    return np.clip(p, EPS, 1 - EPS)


def run_screen(seeds, config, rule):
    flag = has_short_term_debt()
    P, y = {}, None
    for seed in seeds:
        y_seed, Z, _ = _load(CONFIGS[config], seed)
        if y is None:
            y = y_seed
        assert np.array_equal(y, y_seed)
        P[seed] = _blend_oof_p(y, Z, seed, rule)

    print("=== 1. 全体では効く（有無だけ、742行）===")
    tab = pd.DataFrame({"有無": flag.astype(int), "y": y})
    print(tab.groupby("有無").agg(n=("y", "size"), 実購入率=("y", "mean"))
          .rename(index={0: "借入なし", 1: "借入あり"}).round(3).to_string())
    print(f"  全体AUC（借入なし=買う向き） {roc_auc_score(y, 1 - flag):.3f}")

    p = P[seeds[-1] if 42 not in seeds else 42]
    print("\n=== 2. なぜ帯で消えるか: 有無は既に群の割り当てに使われている ===")
    g = pd.cut(p, [0, WIDE[0], WIDE[1], 1.0],
               labels=["確信-非購入", "曖昧帯", "確信-購入"])
    ct = pd.crosstab(g, flag.astype(int))
    ct.columns = ["借入なしn", "借入ありn"]
    ct["借入あり率"] = (ct["借入ありn"] / ct.sum(1)).round(3)
    print(ct.to_string())
    print("\n  群ごとの較正（実測 vs 予測平均）")
    cal = pd.DataFrame({"有無": flag.astype(int), "y": y, "p": p})
    print(cal.groupby("有無").agg(n=("y", "size"), 実測=("y", "mean"),
                                 予測=("p", "mean"))
          .rename(index={0: "借入なし", 1: "借入あり"}).round(3).to_string())

    print("\n=== 3. 帯の中で割れるか（差 = 借入あり率 - 借入なし率）===")
    print(f"{'帯':<12}{'n中央値':>8}{'差の平均':>10}{'差>0のseed':>12}"
          f"{'seed42の差の95%CI':>26}")
    for lo, hi in SCREEN_BANDS:
        diffs, ns = [], []
        for seed in seeds:
            band = (P[seed] > lo) & (P[seed] < hi)
            yy, hh = y[band], flag[band]
            if (hh == 0).sum() < 10 or (hh == 1).sum() < 10:
                continue
            diffs.append(yy[hh == 1].mean() - yy[hh == 0].mean())
            ns.append(int(band.sum()))
        if not diffs:
            continue
        ref = P[42] if 42 in P else P[seeds[0]]
        band = (ref > lo) & (ref < hi)
        yy, hh = y[band], flag[band]
        n0, n1 = max(int((hh == 0).sum()), 1), max(int((hh == 1).sum()), 1)
        r0, r1 = yy[hh == 0].mean(), yy[hh == 1].mean()
        se = np.sqrt(r0 * (1 - r0) / n0 + r1 * (1 - r1) / n1)
        d = r1 - r0
        print(f"{lo}-{hi:<7}{int(np.median(ns)):>8}{np.mean(diffs):>+10.3f}"
              f"{sum(1 for x in diffs if x > 0):>9}/{len(diffs)}"
              f"{f'[{d - 1.96 * se:+.3f}, {d + 1.96 * se:+.3f}]':>26}")
    print("\n  帯を狭めるほど符号が反転して見えるが、その帯は n が 100 を切る。")
    print("  帯幅を7通り試している時点で多重比較なので、strict で決める。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="screen", choices=("screen", "strict"))
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--alpha-rule", default="1se", choices=("auto", "1se"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if args.mode == "screen":
        run_screen(args.seeds, args.config, args.alpha_rule)
    else:
        run_strict(args.seeds, args.config, args.alpha_rule)


if __name__ == "__main__":
    main()
