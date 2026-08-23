"""H2財務健全性4列のE0(木)/E0b(線形)への配分を5seedで比較する。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, E0B_NAME, ORG_SCORE_COL, _lgbm, _nested_text, _scores,
    blend_oof, build_features, fit_e0b,
)
from expert_groups import (  # noqa: E402
    FINANCE_COLS, FINANCE_ISNA_COLS, H1_COLS, SCALE_COLS, SURVEY_COLS,
    SURVEY_ISNA_COLS, e0_anchor_cols,
)
from organization_features import fold_org_preds, load_org_text  # noqa: E402
from text_features import TEXT_COL  # noqa: E402
from company_name_features import add_company_name_features  # noqa: E402
from company_overview_features import add_company_overview_manual_features  # noqa: E402
from compare_pruned_restore_legacy_batch1 import add_h29_structure  # noqa: E402
from dx_outlook_features import add_dx_outlook_priority_features  # noqa: E402
from finance_ratio_features import add_finance_ratios  # noqa: E402
from readiness_to_scale_features import add_readiness_features  # noqa: E402
from workforce_axis_features import add_workforce_axes, add_workforce_interaction  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
H2 = ("自己資本比率", "負債比率", "ROA", "ROE")
PAIR = ("自己資本比率", "ROA")
COMP = ("負債比率", "ROE")
CORE4 = ("LLM_need_score", "LLM_make_limitation_score",
         "LLM_buy_openness_score", "LLM_internal_training_capability_score")
ROUTES = {
    "tree4_linear0": (H2, ()),
    "tree4_linear_complement": (H2, COMP),
    "tree4_linear4": (H2, H2),
    "tree_pair_linear_complement": (PAIR, COMP),
    "tree_complement_linear0": (COMP, ()),
    "anchor_both4": (H2, H2),
    "route_llm_both": (H2, H2),
    "route_dx_both": (H2, H2),
    "route_dx_none": (H2, H2),
    "route_missing_both": (H2, H2),
    "route_missing_none": (H2, H2),
    "route_org_tree_only": (H2, H2),
    "route_org_linear_only": (H2, H2),
    "route_org_anchor_none": (H2, H2),
    "route_h18_tree_only": (H2, H2),
    "route_h18_linear_only": (H2, H2),
    "route_h18_anchor_none": (H2, H2),
    "route_survey_tree_only": (H2, H2),
    "route_survey_linear_only": (H2, H2),
    "route_survey_anchor_none": (H2, H2),
    "route_finance_tree_only": (H2, H2),
    "route_finance_linear_only": (H2, H2),
    "route_finance_anchor_none": (H2, H2),
    "H21_tree_only": (H2, H2),
    "H21_linear_only": (H2, H2),
    "H21_both": (H2, H2),
    "H37_tree_only": (H2, H2),
    "H37_linear_only": (H2, H2),
    "H37_both": (H2, H2),
    "H20_tree_only": (H2, H2),
    "H20_linear_only": (H2, H2),
    "H20_both": (H2, H2),
    "H49_tree_only": (H2, H2),
    "H49_linear_only": (H2, H2),
    "H49_both": (H2, H2),
    "H35_tree_only": (H2, H2),
    "H35_linear_only": (H2, H2),
    "H35_both": (H2, H2),
    "H16_tree_only": (H2, H2),
    "H16_linear_only": (H2, H2),
    "H16_both": (H2, H2),
    "H29_tree_only": (H2, H2),
    "H29_linear_only": (H2, H2),
    "H29_both": (H2, H2),
}

EXTRA_GROUPS = {
    "H21": ["ソフト投資比率_売上", "流動比率"],
    "H37": ["H37_governance_count", "H37_priority_count"],
    "H20": ["企業名_カタカナ比率", "企業名_文字数"],
    "H49": ["展開段階_移行幅", "展開段階_scale_window"],
    "H35": ["人材_推定対象比率", "人材_推定研修対象規模"],
    "H16": ["概要_digital_minus_low", "概要_low_x_physical"],
    "H29": ["組織_ノード数", "組織_最大深さ"],
}


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True, llm_makebuy=True,
                               h2_restore=True)
    for add in (add_finance_ratios, add_dx_outlook_priority_features,
                add_company_name_features, add_readiness_features,
                add_workforce_axes, add_workforce_interaction,
                add_company_overview_manual_features, add_h29_structure):
        X, Xte = add(train, X), add(test, Xte)
    X0, _, _ = build_features(train, test)
    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    all_anchor = e0_anchor_cols(X.columns)
    tree_base = [c for c in all_anchor if c in X0.columns and c not in H2
                 and not c.startswith("DX展望_")
                 and c not in missing and not c.startswith("LLM_")]
    linear_base = list(e0_anchor_cols(X0.columns))
    dx_cols = [c for c in all_anchor if c.startswith("DX展望_")]
    missing_cols = [c for c in all_anchor if c in missing]
    org_cols = [c for c in all_anchor if c.startswith("組織図_")]
    h18_cols = [c for c in all_anchor if c.startswith("概要_")]
    survey_cols = [c for c in all_anchor if c in SURVEY_COLS]
    finance_cols = [c for c in all_anchor if c in FINANCE_COLS + SCALE_COLS + H1_COLS]
    selected = os.environ.get("ROUTING_VARIANTS", "").split(",")
    routes = ({k: ROUTES[k] for k in selected if k}
              if any(selected) else ROUTES)
    txt = train[TEXT_COL].fillna("").astype(str).values
    txt_te = test[TEXT_COL].fillna("").astype(str).values
    org, org_te = load_org_text(train), load_org_text(test)
    rows = []

    for seed in SEEDS:
        d = np.load(f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03_llmcore4e0only.npz")
        base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        po = {v: np.zeros(len(X)) for v in routes}
        pb = {v: np.zeros(len(X)) for v in routes}
        pt = {v: [] for v in routes}
        pbt = {v: [] for v in routes}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            for name, (tree_h2, linear_h2) in routes.items():
                c0 = tree_base + list(tree_h2) + list(CORE4)
                c0b = linear_base + list(linear_h2)
                if name == "route_llm_both":
                    c0b += list(CORE4)
                if name == "route_dx_both":
                    c0 += dx_cols
                if name == "route_dx_none":
                    c0b = [c for c in c0b if c not in dx_cols]
                if name == "route_missing_both":
                    c0 += missing_cols
                if name == "route_missing_none":
                    c0b = [c for c in c0b if c not in missing_cols]
                if name == "route_org_tree_only":
                    c0b = [c for c in c0b if c not in org_cols]
                if name == "route_org_linear_only":
                    c0 = [c for c in c0 if c not in org_cols]
                if name == "route_org_anchor_none":
                    c0 = [c for c in c0 if c not in org_cols]
                    c0b = [c for c in c0b if c not in org_cols]
                if name == "route_h18_tree_only":
                    c0b = [c for c in c0b if c not in h18_cols]
                if name == "route_h18_linear_only":
                    c0 = [c for c in c0 if c not in h18_cols]
                if name == "route_h18_anchor_none":
                    c0 = [c for c in c0 if c not in h18_cols]
                    c0b = [c for c in c0b if c not in h18_cols]
                if name == "route_survey_tree_only":
                    c0b = [c for c in c0b if c not in survey_cols]
                if name == "route_survey_linear_only":
                    c0 = [c for c in c0 if c not in survey_cols]
                if name == "route_survey_anchor_none":
                    c0 = [c for c in c0 if c not in survey_cols]
                    c0b = [c for c in c0b if c not in survey_cols]
                if name == "route_finance_tree_only":
                    c0b = [c for c in c0b if c not in finance_cols]
                if name == "route_finance_linear_only":
                    c0 = [c for c in c0 if c not in finance_cols]
                if name == "route_finance_anchor_none":
                    c0 = [c for c in c0 if c not in finance_cols]
                    c0b = [c for c in c0b if c not in finance_cols]
                for group, extra in EXTRA_GROUPS.items():
                    if name == f"{group}_tree_only":
                        c0 += extra
                    elif name == f"{group}_linear_only":
                        c0b += extra
                    elif name == f"{group}_both":
                        c0 += extra
                        c0b += extra
                Atr, Ava, Ate = X.iloc[tr][c0].copy(), X.iloc[va][c0].copy(), Xte[c0].copy()
                Btr, Bva, Bte = X.iloc[tr][c0b].copy(), X.iloc[va][c0b].copy(), Xte[c0b].copy()
                for col, vals in ((DX_SCORE_COL, (ttr, tva, tte)),
                                  (ORG_SCORE_COL, (otr, ova, ote))):
                    Atr[col], Ava[col], Ate[col] = vals
                    Btr[col], Bva[col], Bte[col] = vals
                po[name][va], q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                pb[name][va], qb = fit_e0b(Btr, y[tr], Bva, Bte, C=0.03)
                pt[name].append(q)
                pbt[name].append(qb)

        for name in routes:
            oof, te = dict(base_oof), dict(base_te)
            oof["E0_anchor"], oof[E0B_NAME] = po[name], pb[name]
            te["E0_anchor"] = np.mean(pt[name], axis=0)
            te[E0B_NAME] = np.mean(pbt[name], axis=0)
            pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
            score = _scores(y, pred)
            rows.append({"seed": seed, "variant": name,
                         "score": {m: float(score[m]) for m in ("auc", "ap", "f1")}})
            print(seed, name, rows[-1]["score"], flush=True)

    summary = {}
    reference = os.environ.get("ROUTING_REFERENCE", "tree4_linear_complement")
    for name, route in routes.items():
        rs = [r for r in rows if r["variant"] == name]
        refs = [r for r in rows if r["variant"] == reference]
        summary[name] = {"tree": list(route[0]), "linear": list(route[1])}
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["score"][metric] for r in rs])
            ref = np.array([r["score"][metric] for r in refs])
            summary[name][metric] = {"mean": float(vals.mean()),
                                     "delta_vs_asym": float((vals-ref).mean()),
                                     "wins_vs_asym": int((vals > ref).sum())}
    out = os.environ.get("ROUTING_OUTPUT", "exp/result_h2_expert_routing.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
