"""LLM Make-or-Buy特徴によるエキスパート内部の部分差し替え比較。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import (  # noqa: E402
    BASE_PARAMS, DX_SCORE_COL, E0B_NAME, THS, _lgbm, _nested_text,
    _scores, blend_oof, build_features, fit_e0b,
)
from expert_groups import e0_anchor_cols, e6_manual_cols  # noqa: E402
from expert_groups import FINANCE_ISNA_COLS, SURVEY_ISNA_COLS  # noqa: E402
from organization_features import fold_org_preds, load_org_text, ORG_SCORE_COL  # noqa: E402
from text_features import TEXT_COL  # noqa: E402
from compare_llm_make_or_buy import BASE_EXPERTS, load_llm_features  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
VARIANTS = (
    "e0_drop_dx_and_missing",
    "e0_replace_dx_core4_drop_missing",
    "e0_replace_dx_core_unknown_drop_missing",
    "e0_replace_dx_core_stage_drop_missing",
)


def _llm_frames():
    tr, te, cols = load_llm_features("products")
    # fold外で学習を行わない決定的な補完。欠損3件は専用フラグを持つ。
    imp = SimpleImputer(strategy="constant", fill_value=-1)
    return (pd.DataFrame(imp.fit_transform(tr), columns=cols, index=tr.index),
            pd.DataFrame(imp.transform(te), columns=cols, index=te.index), cols)


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    L, Lte, lcols = _llm_frames()
    llm_cols = [f"LLM_{c}" for c in lcols]
    L.columns, Lte.columns = llm_cols, llm_cols
    X = pd.concat([X.reset_index(drop=True), L], axis=1)
    Xte = pd.concat([Xte.reset_index(drop=True), Lte], axis=1)
    base_c0 = e0_anchor_cols([c for c in X.columns if c not in llm_cols])
    base_c6 = e6_manual_cols(X.columns)
    dx_manual = [c for c in base_c0 if c.startswith("DX展望_")]
    dx_manual_e6 = [c for c in base_c6 if c.startswith("DX展望_")]
    txt = train[TEXT_COL].fillna("").astype(str).values
    txt_te = test[TEXT_COL].fillna("").astype(str).values
    org, org_te = load_org_text(train), load_org_text(test)
    all_rows = []

    for seed in SEEDS:
        cache = np.load(f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03.npz")
        base_oof = {n: cache[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: cache[f"te_{n}"] for n in BASE_EXPERTS}
        base_blend, _, _, _ = blend_oof(y, base_oof, base_te, seed,
                                        experts=BASE_EXPERTS)
        base_score = _scores(y, base_blend)
        changed_oof = {v: {} for v in VARIANTS}
        changed_te = {v: {} for v in VARIANTS}
        for v in VARIANTS:
            changed_oof[v] = {"E0_anchor": np.zeros(len(X)),
                              E0B_NAME: np.zeros(len(X))}
            changed_te[v] = {"E0_anchor": [], E0B_NAME: []}

        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            score_values = ((DX_SCORE_COL, (ttr, tva, tte)),
                            (ORG_SCORE_COL, (otr, ova, ote)))
            for variant in VARIANTS:
                missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
                cols = [c for c in base_c0
                        if c not in dx_manual and c not in missing]
                core4 = [f"LLM_{c}" for c in (
                    "need_score", "make_limitation_score", "buy_openness_score",
                    "internal_training_capability_score")]
                unknown = [f"LLM_{c}" for c in (
                    "need_unknown", "make_limitation_unknown",
                    "buy_openness_unknown", "internal_training_capability_unknown",
                    "llm_row_missing")]
                if "core4" in variant:
                    cols += core4
                elif "core_unknown" in variant:
                    cols += core4 + unknown
                elif "core_stage" in variant:
                    cols += core4 + ["LLM_implementation_stage_ordinal"]
                Atr, Ava, Ate = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy(), Xte[cols].copy()
                for col, vals in score_values:
                    if col == DX_SCORE_COL and "score" in variant:
                        continue
                    Atr[col], Ava[col], Ate[col] = vals
                p, q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                changed_oof[variant]["E0_anchor"][va] = p
                changed_te[variant]["E0_anchor"].append(q)
                p, q = fit_e0b(Atr, y[tr], Ava, Ate, C=0.03)
                changed_oof[variant][E0B_NAME][va] = p
                changed_te[variant][E0B_NAME].append(q)

        for variant in VARIANTS:
            oof, te = dict(base_oof), dict(base_te)
            for name, pred in changed_oof[variant].items():
                oof[name] = pred
                te[name] = np.mean(changed_te[variant][name], axis=0)
            pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
            s = _scores(y, pred)
            delta = {k: float(s[k] - base_score[k]) for k in ("auc", "ap", "f1")}
            all_rows.append({"seed": seed, "variant": variant, "delta": delta})
            print(seed, variant, delta)

    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        rows = [r for r in all_rows if r["variant"] == variant]
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["delta"][metric] for r in rows])
            summary[variant][metric] = {"mean": float(vals.mean()),
                                        "wins": int((vals > 0).sum()), "n": 5}
    out = {"seeds": list(SEEDS), "llm_features": llm_cols,
           "rows": all_rows, "summary": summary}
    with open("exp/result_e0_reduced_llm_budget.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
