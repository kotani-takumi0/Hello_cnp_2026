"""H88: search every way of splitting the eight experts across two stages.

H87 tested one split by hand (survey to stage 2, everything else to stage 1) and
lost to the additive meta.  This searches all 254 partitions instead of assuming
that split was the right one.

Selecting the best of 254 on the same seeds that score it is selection on the
evaluation set, so the run is in two phases:

  search   seeds 0-4   -- all 254 partitions, alpha pinned to 0.001
  confirm  seeds 5-9,42 -- the finalists only, with the real select_alpha_1se

The confirm seeds are never looked at during the search, so their numbers are
honest.  A finalist that only shines on the search seeds is exactly what this
design is meant to expose.

Alpha is pinned during the search purely for speed: select_alpha_1se costs 35
meta fits per call and the 1-SE rule lands on the grid bottom in 19 of 20 cases
(meta_native_and_alpha_selection_20260829.md section 3.3).  The confirm phase
uses the real rule.

  logit_final = logit(p1) + λ·(logit(p2) - centre)   inside stage 1's band
              = logit(p1)                            outside

λ comes from a grid containing 0, chosen on the fold's training band.

Run:
  .venv/bin/python exp/search_cascade_split.py
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, EXPERTS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from meta_blend import EPS, fit_meta, select_alpha_1se  # noqa: E402

BAND_LO, BAND_HI = 0.1, 0.9
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 1.5)
SEARCH_SEEDS = (0, 1, 2, 3, 4)
CONFIRM_SEEDS = (5, 6, 7, 8, 9, 42)
PINNED_ALPHA = 0.001
SHORT = {"E0_anchor": "E0", "E1_finance": "E1", "E2_survey": "E2",
         "E3_dx_text": "E3", "E4_org_text": "E4", "E6_manual": "E6",
         "E7_cross": "E7", "E0b_linear": "E0b"}


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _best_f1(y, p):
    return max(f1_score(y, (p >= t).astype(int)) for t in THS)


def _oof_and_apply(Z, y, tr, va, seed, alpha):
    """fold train内の交差適合ロジットと、fold validへの適用を返す。"""
    inner = np.zeros(len(tr))
    for a, b in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z[tr], y[tr]):
        w, c = fit_meta(Z[tr][a], y[tr][a], alpha=alpha)
        inner[b] = Z[tr][b] @ w + c
    w, c = fit_meta(Z[tr], y[tr], alpha=alpha)
    return inner, Z[va] @ w + c


def _cascade(y, Z, stage1, stage2, seed, alpha_rule):
    """1seedぶんの合成OOFを返す。stage1/stage2 は列番号のリスト。"""
    out = np.zeros(len(y))
    lambdas = []
    for tr, va in StratifiedKFold(
            5, shuffle=True, random_state=seed).split(Z, y):
        a1 = (select_alpha_1se(Z[tr][:, stage1], y[tr], seed)
              if alpha_rule == "1se" else PINNED_ALPHA)
        inner1, apply1 = _oof_and_apply(Z[:, stage1], y, tr, va, seed, a1)
        if not stage2:
            out[va] = expit(apply1)
            lambdas.append(0.0)
            continue
        a2 = (select_alpha_1se(Z[tr][:, stage2], y[tr], seed)
              if alpha_rule == "1se" else PINNED_ALPHA)
        inner2, apply2 = _oof_and_apply(Z[:, stage2], y, tr, va, seed, a2)

        band_tr = (expit(inner1) > BAND_LO) & (expit(inner1) < BAND_HI)
        centre = float(np.median(inner2))
        best, best_ap = 0.0, -np.inf
        if band_tr.sum() >= 30 and len(np.unique(y[tr][band_tr])) > 1:
            for lam in LAMBDA_GRID:
                score = inner1[band_tr] + lam * (inner2[band_tr] - centre)
                ap = average_precision_score(y[tr][band_tr], score)
                if ap > best_ap:
                    best, best_ap = lam, ap
        band_va = (expit(apply1) > BAND_LO) & (expit(apply1) < BAND_HI)
        score = apply1.copy()
        score[band_va] += best * (apply2[band_va] - centre)
        out[va] = expit(score)
        lambdas.append(best)
    return out, float(np.mean(lambdas))


def _score(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": _best_f1(y, p)}


def _label(stage1, stage2):
    s1 = "+".join(SHORT[EXPERTS[i]] for i in stage1)
    s2 = "+".join(SHORT[EXPERTS[i]] for i in stage2) if stage2 else "-"
    return f"{s1} → {s2}"


def main(config, top_k):
    data = {s: _load(CONFIGS[config], s)[:2] for s in SEARCH_SEEDS + CONFIRM_SEEDS}
    all_columns = list(range(len(EXPERTS)))

    partitions = []
    for size in range(1, len(EXPERTS)):
        for stage2 in itertools.combinations(all_columns, size):
            stage1 = [i for i in all_columns if i not in stage2]
            partitions.append((stage1, list(stage2)))
    print(f"分け方 {len(partitions)} 通りを seed {SEARCH_SEEDS} で探索")

    rows = []
    for n, (stage1, stage2) in enumerate(partitions, 1):
        if n % 50 == 0:
            print(f"  {n}/{len(partitions)}", flush=True)
        scores, lams = [], []
        for seed in SEARCH_SEEDS:
            y, Z = data[seed]
            p, lam = _cascade(y, Z, stage1, stage2, seed, "pinned")
            scores.append(_score(y, p))
            lams.append(lam)
        rows.append({
            "label": _label(stage1, stage2),
            "stage1": tuple(stage1), "stage2": tuple(stage2),
            "n_stage2": len(stage2), "lambda": float(np.mean(lams)),
            **{k: float(np.mean([s[k] for s in scores]))
               for k in ("auc", "ap", "f1")},
        })
    search = pd.DataFrame(rows).sort_values("ap", ascending=False)

    baseline = []
    for seed in SEARCH_SEEDS:
        y, Z = data[seed]
        p, _ = _cascade(y, Z, all_columns, [], seed, "pinned")
        baseline.append(_score(y, p))
    base_ap = float(np.mean([s["ap"] for s in baseline]))
    print(f"\n探索seedでの full_meta（1段のみ）AP = {base_ap:.5f}")
    print(f"\n=== 探索上位{top_k}（seed 0-4 の平均、この順位は選択バイアス込み）===")
    print(search.head(top_k)[
        ["label", "n_stage2", "lambda", "auc", "ap", "f1"]].to_string(
            index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\n  full_meta を上回った分け方: "
          f"{int((search['ap'] > base_ap).sum())}/{len(search)}")

    finalists = search.head(top_k)
    print(f"\n=== 検証（seed {CONFIRM_SEEDS}、探索に一度も使っていない）===")
    confirm_rows = []
    for seed in CONFIRM_SEEDS:
        y, Z = data[seed]
        p, _ = _cascade(y, Z, all_columns, [], seed, "1se")
        confirm_rows.append({"seed": seed, "label": "full_meta（現行）",
                             **_score(y, p)})
        for _, r in finalists.iterrows():
            p, _ = _cascade(y, Z, list(r.stage1), list(r.stage2), seed, "1se")
            confirm_rows.append({"seed": seed, "label": r.label, **_score(y, p)})
    confirm = pd.DataFrame(confirm_rows)
    print(confirm.groupby("label")[["auc", "ap", "f1"]].mean().sort_values(
        "ap", ascending=False).to_string(float_format=lambda v: f"{v:.5f}"))

    base = confirm[confirm.label == "full_meta（現行）"].sort_values(
        "seed").reset_index(drop=True)
    print("\n=== full_meta を基準にしたペア差（検証seed）===")
    for label in finalists.label:
        cand = confirm[confirm.label == label].sort_values(
            "seed").reset_index(drop=True)
        line = [f"  {label:28s}"]
        for metric in ("auc", "ap", "f1"):
            d = cand[metric].values - base[metric].values
            line.append(f"{metric} {d.mean():+.5f} ({int((d > 0).sum())}/{len(d)})")
        decision, _ = verdict(base, cand)
        print("  ".join(line) + f"   {decision}")

    search.to_csv("exp/cascade_split_search.csv", index=False)
    confirm.to_csv("exp/cascade_split_confirm.csv", index=False)
    print("\nsaved: exp/cascade_split_search.csv, exp/cascade_split_confirm.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    main(args.config, args.top_k)
