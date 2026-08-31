"""H95: where should E2 put the companies that have no process tool at all?

`アンケート７` (satisfaction with the existing process-improvement tool) is
asked only of companies that answered "yes" to `アンケート６`.  For the other
356 companies the answer is missing, and `dataset.ZERO_FILL` imputes **0** --
below "非常に低い"(1) on a 1-5 satisfaction scale.  That is a statement that
non-adopters are more dissatisfied than the most dissatisfied adopter.

The repository's own `survey_features.py` disagrees with itself here: it uses
`_UNINSTALLED_A7 = 3.3`, derived from the data (non-adopters buy at 0.222,
which sits between a7=3 at 0.247 and a7=4 at 0.169).  E2 was the one place
still using the sentinel.

The fix touches E2 only -- E0/E0b/E7 keep seeing the 0-fill, so their recorded
values stay reproducible.  `--a7-fill` in `ensemble_experts.py` is off by
default for the same reason.

This script compares the two caches over the production outer folds, under
both alpha rules, because production submits with `--alpha-rule 1se`.

Build the candidate caches first:

    for s in 0 1 2 3 4 5 6 7 8 9 42; do
      .venv/bin/python exp/ensemble_experts.py --seed $s --concat-embed \
        --cross --e7-model lr --linear --linear-c 0.03 --h2h7-both4 \
        --a7-fill 3.3
    done

Then:

    .venv/bin/python exp/compare_survey_a7_fill.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import verdict  # noqa: E402
from ensemble_experts import THS, blend_oof  # noqa: E402

EXPERTS = ("E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear")
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
BASE_CACHE = "exp/_experts_seed{seed}_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
CAND_CACHE = BASE_CACHE.replace(".npz", "_a7fill{fill:g}.npz")


def _load(path):
    d = np.load(path, allow_pickle=True)
    return (d["y"], {n: d["oof_" + n] for n in EXPERTS},
            {n: d["te_" + n] for n in EXPERTS})


def _score(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS)}


def main(seeds, fill, rules):
    rows = []
    for rule in rules:
        for seed in seeds:
            y, oof, te = _load(BASE_CACHE.format(seed=seed))
            yc, oofc, tec = _load(CAND_CACHE.format(seed=seed, fill=fill))
            assert np.array_equal(y, yc)
            for name in EXPERTS:
                if name == "E2_survey":
                    continue
                assert np.allclose(oof[name], oofc[name]), (
                    f"{name} が変わっている。E2以外は不変のはず")
            base, _, _, _ = blend_oof(y, oof, te, seed=seed,
                                      experts=list(EXPERTS), alpha_rule=rule)
            cand, _, _, _ = blend_oof(yc, oofc, tec, seed=seed,
                                      experts=list(EXPERTS), alpha_rule=rule)
            rows.append({"rule": rule, "seed": seed, "arm": "base",
                         "e2": roc_auc_score(y, oof["E2_survey"]),
                         **_score(y, base)})
            rows.append({"rule": rule, "seed": seed, "arm": "a7fill",
                         "e2": roc_auc_score(y, oofc["E2_survey"]),
                         **_score(y, cand)})
    r = pd.DataFrame(rows)

    print(f"=== E2の a7 埋め値 0 → {fill:g}（本番の外側5fold、{len(seeds)}seed）===\n")
    print(f"{'alpha規則':<10}{'E2単体AUC':>20}{'ΔAUC':>10}{'ΔAP':>10}"
          f"{'ΔF1':>10}{'ΔAP>0':>8}  判定")
    for rule in rules:
        sub = r[r.rule == rule]
        b = sub[sub.arm == "base"].sort_values("seed").reset_index(drop=True)
        c = sub[sub.arm == "a7fill"].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{rule:<10}{f'{b.e2.mean():.4f} -> {c.e2.mean():.4f}':>20}"
              f"{(c.auc - b.auc).mean():>+10.4f}{(c.ap - b.ap).mean():>+10.4f}"
              f"{(c.f1 - b.f1).mean():>+10.4f}"
              f"{int((c.ap > b.ap).sum()):>6}/{len(c)}  {v}")
    r.to_csv("exp/survey_a7_fill_verdict.csv", index=False)
    print("\nsaved: exp/survey_a7_fill_verdict.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--fill", type=float, default=3.3)
    ap.add_argument("--rules", nargs="+", default=["auto", "1se"],
                    choices=["auto", "1se"])
    args = ap.parse_args()
    main(args.seeds, args.fill, args.rules)
