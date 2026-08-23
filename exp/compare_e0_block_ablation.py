"""E0/E0b内部の情報ブロックを削り、冗長な入力を特定する。"""
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
from expert_groups import (  # noqa: E402
    CAT_COLS, FINANCE_COLS, FINANCE_ISNA_COLS, H1_COLS, SCALE_COLS,
    SURVEY_COLS, SURVEY_ISNA_COLS, e0_anchor_cols,
)
from organization_features import ORG_SCORE_COL, fold_org_preds, load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)


def block_map(columns):
    return {
        "drop_dx_manual": [c for c in columns if c.startswith("DX展望_")],
        "drop_org_manual": [c for c in columns if c.startswith("組織図_")],
        "drop_overview_manual": [c for c in columns if c.startswith("概要_")],
        "drop_finance": [c for c in FINANCE_COLS + H1_COLS if c in columns],
        "drop_scale": [c for c in SCALE_COLS if c in columns],
        "drop_survey": [c for c in SURVEY_COLS if c in columns],
        "drop_categories": [c for c in CAT_COLS if c in columns],
        "drop_missing_flags": [c for c in FINANCE_ISNA_COLS + SURVEY_ISNA_COLS
                               if c in columns],
    }


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    base_cols = e0_anchor_cols(X.columns)
    blocks = block_map(base_cols)
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
        preds = {v: {"E0_anchor": np.zeros(len(X)), E0B_NAME: np.zeros(len(X))}
                 for v in blocks}
        tests = {v: {"E0_anchor": [], E0B_NAME: []} for v in blocks}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            for variant, removed in blocks.items():
                cols = [c for c in base_cols if c not in removed]
                Atr, Ava, Ate = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy(), Xte[cols].copy()
                for col, values in ((DX_SCORE_COL, (ttr, tva, tte)),
                                    (ORG_SCORE_COL, (otr, ova, ote))):
                    Atr[col], Ava[col], Ate[col] = values
                p, q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                preds[variant]["E0_anchor"][va] = p
                tests[variant]["E0_anchor"].append(q)
                p, q = fit_e0b(Atr, y[tr], Ava, Ate, C=0.03)
                preds[variant][E0B_NAME][va] = p
                tests[variant][E0B_NAME].append(q)
        for variant in blocks:
            oof, te = dict(base_oof), dict(base_te)
            for name in ("E0_anchor", E0B_NAME):
                oof[name] = preds[variant][name]
                te[name] = np.mean(tests[variant][name], axis=0)
            pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
            s = _scores(y, pred)
            delta = {k: float(s[k] - sb[k]) for k in ("auc", "ap", "f1")}
            rows.append({"seed": seed, "variant": variant,
                         "removed_columns": len(blocks[variant]), "delta": delta})
            print(seed, variant, len(blocks[variant]), delta)
    summary = {}
    for variant, removed in blocks.items():
        rs = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"removed_columns": removed, "n_removed": len(removed)}
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["delta"][metric] for r in rs])
            summary[variant][metric] = {"mean": float(vals.mean()),
                                        "wins": int((vals > 0).sum()), "n": 5}
    result = {"seeds": list(SEEDS), "rows": rows, "summary": summary}
    with open("exp/result_e0_block_ablation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
