"""LLM親4軸のablationと、E0/E0bへの適用位置を比較する。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, E0B_NAME, _lgbm, _nested_text, _scores, blend_oof,
    build_features, fit_e0b,
)
from expert_groups import FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols  # noqa: E402
from llm_make_or_buy_features import add_llm_make_or_buy_features  # noqa: E402
from organization_features import ORG_SCORE_COL, fold_org_preds, load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
AXES = (
    "LLM_need_score", "LLM_make_limitation_score", "LLM_buy_openness_score",
    "LLM_internal_training_capability_score",
)
SUBSETS = {"core4": list(AXES)}
SUBSETS.update({f"without_{c.removeprefix('LLM_').removesuffix('_score')}":
                [x for x in AXES if x != c] for c in AXES})
PLACEMENTS = ("e0_only", "e0b_only", "both")


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X = add_llm_make_or_buy_features(train, X, "train")
    Xte = add_llm_make_or_buy_features(test, Xte, "test")
    base_cols = e0_anchor_cols(X.columns)
    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    retained = [c for c in base_cols if not c.startswith("DX展望_")
                and c not in missing and not c.startswith("LLM_")]
    txt = train[TEXT_COL].fillna("").astype(str).values
    txt_te = test[TEXT_COL].fillna("").astype(str).values
    org, org_te = load_org_text(train), load_org_text(test)
    rows = []

    for seed in SEEDS:
        d = np.load(f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03.npz")
        base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        base, _, _, _ = blend_oof(y, base_oof, base_te, seed,
                                  experts=BASE_EXPERTS)
        sb = _scores(y, base)
        po = {s: {"E0_anchor": np.zeros(len(X)), E0B_NAME: np.zeros(len(X))}
              for s in SUBSETS}
        pt = {s: {"E0_anchor": [], E0B_NAME: []} for s in SUBSETS}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            for subset, axes in SUBSETS.items():
                cols = retained + axes
                Atr, Ava, Ate = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy(), Xte[cols].copy()
                for col, vals in ((DX_SCORE_COL, (ttr, tva, tte)),
                                  (ORG_SCORE_COL, (otr, ova, ote))):
                    Atr[col], Ava[col], Ate[col] = vals
                p, q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                po[subset]["E0_anchor"][va] = p
                pt[subset]["E0_anchor"].append(q)
                p, q = fit_e0b(Atr, y[tr], Ava, Ate, C=0.03)
                po[subset][E0B_NAME][va] = p
                pt[subset][E0B_NAME].append(q)
        for subset in SUBSETS:
            for placement in PLACEMENTS:
                oof, te = dict(base_oof), dict(base_te)
                changed = (("E0_anchor",) if placement == "e0_only" else
                           (E0B_NAME,) if placement == "e0b_only" else
                           ("E0_anchor", E0B_NAME))
                for name in changed:
                    oof[name] = po[subset][name]
                    te[name] = np.mean(pt[subset][name], axis=0)
                pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
                s = _scores(y, pred)
                delta = {k: float(s[k] - sb[k]) for k in ("auc", "ap", "f1")}
                rows.append({"seed": seed, "subset": subset,
                             "placement": placement, "n_axes": len(SUBSETS[subset]),
                             "delta": delta})
                print(seed, subset, placement, delta)
    summary = {}
    for subset in SUBSETS:
        for placement in PLACEMENTS:
            key = f"{subset}__{placement}"
            rs = [r for r in rows if r["subset"] == subset and
                  r["placement"] == placement]
            summary[key] = {"n_axes": len(SUBSETS[subset])}
            for metric in ("auc", "ap", "f1"):
                vals = np.array([r["delta"][metric] for r in rs])
                summary[key][metric] = {"mean": float(vals.mean()),
                                        "wins": int((vals > 0).sum()), "n": 5}
    with open("exp/result_e0_core_axis_ablation.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    ranked = sorted(summary.items(), key=lambda x: x[1]["ap"]["mean"], reverse=True)
    print(json.dumps(ranked[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
