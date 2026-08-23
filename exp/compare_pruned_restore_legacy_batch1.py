"""過去REJECTの表形式特徴を、情報削減後のE0へ少量ずつ戻して再審査する。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, _lgbm, _nested_text, _scores, blend_oof, build_features,
)
from expert_groups import FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols  # noqa: E402
from finance_ratio_features import add_finance_ratios  # noqa: E402
from company_name_features import add_company_name_features  # noqa: E402
from company_overview_features import add_company_overview_manual_features  # noqa: E402
from dx_outlook_features import add_dx_outlook_priority_features  # noqa: E402
from hypotheses import (  # noqa: E402
    h2_financial_health, h3_digital_density, h4_need_gap,
    h4b_need_gap_v2, h7_cash_slack,
)
from llm_make_or_buy_features import add_llm_make_or_buy_features  # noqa: E402
from organization_features import ORG_SCORE_COL, fold_org_preds, load_org_text  # noqa: E402
from readiness_to_scale_features import add_readiness_features  # noqa: E402
from survey_features import add_survey_features  # noqa: E402
from survey_purchase_state_features import add_survey_purchase_states  # noqa: E402
from text_features import TEXT_COL  # noqa: E402
from workforce_axis_features import add_workforce_axes, add_workforce_interaction  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
CORE4 = ["LLM_need_score", "LLM_make_limitation_score",
         "LLM_buy_openness_score", "LLM_internal_training_capability_score"]
RESTORES = {
    "H2_financial_health": ["自己資本比率", "ROA"],
    "H2_equity_ratio_only": ["自己資本比率"],
    "H2_roa_only": ["ROA"],
    "H3_digital_density": ["ソフト投資_per従業員", "売上_per従業員"],
    "H4_need_gap": ["不満スコア", "戦略_満足ギャップ"],
    "H4b_dx_gap": ["DXギャップ", "ツール不満"],
    "H7_cash_slack": ["FCF比率", "現金ROA"],
    "H7_fcf_only": ["FCF比率"],
    "H7_cash_roa_only": ["現金ROA"],
    "H2_H7_combined": ["自己資本比率", "ROA", "FCF比率", "現金ROA"],
    "H21_finance_ratio": ["ソフト投資比率_売上", "流動比率"],
    "H22_survey_step": ["アンケ_明確高", "アンケ_連携高"],
    "H2_plus_cash_roa": ["自己資本比率", "ROA", "現金ROA"],
    "H2_plus_H49": ["自己資本比率", "ROA", "展開段階_移行幅", "展開段階_scale_window"],
    "H2_plus_H21": ["自己資本比率", "ROA", "ソフト投資比率_売上", "流動比率"],
    "H2_plus_software_sales": ["自己資本比率", "ROA", "ソフト投資比率_売上"],
    "H2_plus_current_ratio": ["自己資本比率", "ROA", "流動比率"],
    "H2_plus_cash_roa_H49": ["自己資本比率", "ROA", "現金ROA",
                              "展開段階_移行幅", "展開段階_scale_window"],
    "H2_plus_H14": ["自己資本比率", "ROA", "DX展望_future_education_count",
                     "DX展望_vendor_x_education"],
    "H2_plus_H16": ["自己資本比率", "ROA", "概要_digital_minus_low",
                     "概要_low_x_physical"],
    "H2_plus_H20": ["自己資本比率", "ROA", "企業名_カタカナ比率", "企業名_文字数"],
    "H2_plus_H29": ["自己資本比率", "ROA", "組織_ノード数", "組織_最大深さ"],
    "H2_plus_H35": ["自己資本比率", "ROA", "人材_推定対象比率", "人材_推定研修対象規模"],
    "H2_plus_H37": ["自己資本比率", "ROA", "H37_governance_count", "H37_priority_count"],
    "H2_without_H18": ["自己資本比率", "ROA"],
    "H2_plus_state_intent_execution": ["自己資本比率", "ROA", "状態_高意欲_低実行"],
    "H2_plus_state_adopted_dissatisfied": ["自己資本比率", "ROA", "状態_導入済み_低満足"],
    "H2_plus_state_problem_external": ["自己資本比率", "ROA", "状態_高課題_高外部連携"],
    "H2_plus_state_lowmake_highbuy": ["自己資本比率", "ROA", "状態_低内製_高外部調達"],
    "H2_plus_survey_states4": ["自己資本比率", "ROA", "状態_高意欲_低実行",
                               "状態_導入済み_低満足", "状態_高課題_高外部連携",
                               "状態_低内製_高外部調達"],
}
H18_COLS = {
    "概要_high_dx", "概要_high_iot", "概要_high_it_ict",
    "概要_low_construction", "概要_low_education_training", "概要_low_medical",
    "概要_high3_count", "概要_low3_count", "概要_high3_minus_low3",
}


def add_h29_structure(raw, frame):
    """H29記録にあるノード数と最大深さを組織図の罫線から復元する。"""
    out = frame.copy()
    nodes, depths = [], []
    for text in raw["組織図"].fillna("").astype(str):
        lines = [line for line in text.splitlines() if line.strip()]
        node_lines = [line for line in lines if "├" in line or "└" in line]
        depth = [line.count("│") + (1 if "├" in line or "└" in line else 0)
                 for line in lines]
        nodes.append(len(node_lines))
        depths.append(max(depth, default=0))
    out["組織_ノード数"] = nodes
    out["組織_最大深さ"] = depths
    return out


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X = add_llm_make_or_buy_features(train, X, "train")
    Xte = add_llm_make_or_buy_features(test, Xte, "test")
    adders = (
        h2_financial_health, h3_digital_density, h4_need_gap,
        h4b_need_gap_v2, h7_cash_slack, add_finance_ratios,
        add_readiness_features,
        add_company_overview_manual_features, add_company_name_features,
        add_h29_structure, add_workforce_axes, add_workforce_interaction,
        add_dx_outlook_priority_features,
        add_survey_purchase_states,
        lambda raw, frame: add_survey_features(raw, frame, groups=("step",)),
    )
    for add in adders:
        X, Xte = add(train, X), add(test, Xte)

    base_x = build_features(train, test)[0]
    original = e0_anchor_cols(base_x.columns)
    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    retained = [c for c in original if not c.startswith("DX展望_") and c not in missing]
    txt, txt_te = (train[TEXT_COL].fillna("").astype(str).values,
                   test[TEXT_COL].fillna("").astype(str).values)
    org, org_te = load_org_text(train), load_org_text(test)
    selected = os.environ.get("RESTORE_VARIANTS", "").split(",")
    restores = ({k: RESTORES[k] for k in selected if k}
                if any(selected) else RESTORES)
    rows = []

    for seed in SEEDS:
        cache = f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03_llmcore4e0only.npz"
        d = np.load(cache)
        base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        base, _, _, _ = blend_oof(y, base_oof, base_te, seed, experts=BASE_EXPERTS)
        base_score = _scores(y, base)
        preds = {name: np.zeros(len(X)) for name in restores}
        tests = {name: [] for name in restores}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)

        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            for name, extra in restores.items():
                cols = retained + CORE4 + extra
                if name == "H2_without_H18":
                    cols = [c for c in cols if c not in H18_COLS]
                Atr, Ava, Ate = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy(), Xte[cols].copy()
                Atr[DX_SCORE_COL], Ava[DX_SCORE_COL], Ate[DX_SCORE_COL] = ttr, tva, tte
                Atr[ORG_SCORE_COL], Ava[ORG_SCORE_COL], Ate[ORG_SCORE_COL] = otr, ova, ote
                p, q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                preds[name][va] = p
                tests[name].append(q)

        for name in restores:
            oof, te = dict(base_oof), dict(base_te)
            oof["E0_anchor"] = preds[name]
            te["E0_anchor"] = np.mean(tests[name], axis=0)
            pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
            score = _scores(y, pred)
            delta = {m: float(score[m] - base_score[m]) for m in ("auc", "ap", "f1")}
            rows.append({"seed": seed, "variant": name, "delta": delta})
            print(seed, name, delta, flush=True)

    summary = {}
    for name, cols in restores.items():
        rs = [r for r in rows if r["variant"] == name]
        summary[name] = {"columns": cols}
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["delta"][metric] for r in rs])
            summary[name][metric] = {
                "mean": float(vals.mean()), "wins": int((vals > 0).sum()), "n": len(vals)
            }
    out = os.environ.get("RESTORE_OUTPUT", "exp/result_pruned_restore_legacy_batch1.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
