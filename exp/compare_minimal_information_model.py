"""LLM親軸から始め、必要情報だけを段階追加する最小購買モデル。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import _scores, blend_oof, build_features  # noqa: E402
from llm_make_or_buy_features import add_llm_make_or_buy_features  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
LEAN_NAME = "E10_minimal"
CORE = [
    "LLM_need_score", "LLM_make_limitation_score", "LLM_buy_openness_score",
    "LLM_internal_training_capability_score",
    "LLM_implementation_stage_ordinal", "LLM_need_unknown",
    "LLM_make_limitation_unknown", "LLM_buy_openness_unknown",
    "LLM_internal_training_capability_unknown", "LLM_llm_row_missing",
]
FINANCE = [
    "従業員数", "事業所数", "売上", "営業利益率", "営業CF", "投資CF",
    "無形固定資産変動(ソフトウェア関連)", "自己資本",
]
SURVEY = ["アンケート２", "アンケート４", "アンケート７", "アンケート１０"]
STAGES = {
    "core10": CORE,
    "core_fin18": CORE + FINANCE,
    "core_fin_survey22": CORE + FINANCE + SURVEY,
    "core_fin_survey_org23": CORE + FINANCE + SURVEY + ["STACK_E4"],
    "core_fin_survey_org_dx24": CORE + FINANCE + SURVEY + ["STACK_E4", "STACK_E3"],
}


def fit_oof(seed, y, X, Xte):
    oof, parts = np.zeros(len(X)), []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.03, solver="liblinear", max_iter=3000,
                               random_state=seed),
        )
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        parts.append(model.predict_proba(Xte)[:, 1])
    return oof, np.mean(parts, axis=0)


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X = add_llm_make_or_buy_features(train, X, "train")
    Xte = add_llm_make_or_buy_features(test, Xte, "test")
    rows = []
    lean_experts = tuple(n for n in BASE_EXPERTS if n not in
                         ("E0_anchor", "E0b_linear")) + (LEAN_NAME,)

    for seed in SEEDS:
        d = np.load(f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03.npz")
        base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        base, _, _, _ = blend_oof(y, base_oof, base_te, seed,
                                  experts=BASE_EXPERTS)
        base_score = _scores(y, base)
        for stage, cols in STAGES.items():
            A, Ate = X.copy(), Xte.copy()
            A["STACK_E4"], Ate["STACK_E4"] = d["oof_E4_org_text"], d["te_E4_org_text"]
            A["STACK_E3"], Ate["STACK_E3"] = d["oof_E3_dx_text"], d["te_E3_dx_text"]
            p, q = fit_oof(seed, y, A[cols], Ate[cols])
            oof, te = dict(base_oof), dict(base_te)
            oof[LEAN_NAME], te[LEAN_NAME] = p, q
            lean, _, _, _ = blend_oof(y, oof, te, seed, experts=lean_experts)
            ss, sl = _scores(y, p), _scores(y, lean)
            rows.append({
                "seed": seed, "stage": stage, "n_columns": len(cols),
                "standalone": {k: float(ss[k]) for k in ("auc", "ap", "f1")},
                "lean_ensemble": {k: float(sl[k]) for k in ("auc", "ap", "f1")},
                "delta_vs_full8": {k: float(sl[k] - base_score[k])
                                   for k in ("auc", "ap", "f1")},
            })
            print(seed, stage, "single", {k: round(ss[k], 4) for k in ("auc","ap","f1")},
                  "lean-full8", {k: round(sl[k]-base_score[k], 4) for k in ("auc","ap","f1")})
    summary = {}
    for stage in STAGES:
        rs = [r for r in rows if r["stage"] == stage]
        summary[stage] = {}
        for group in ("standalone", "lean_ensemble", "delta_vs_full8"):
            summary[stage][group] = {}
            for metric in ("auc", "ap", "f1"):
                vals = np.array([r[group][metric] for r in rs])
                summary[stage][group][metric] = {
                    "mean": float(vals.mean()),
                    "wins": int((vals > 0).sum()) if group == "delta_vs_full8" else None,
                }
    out = {"seeds": list(SEEDS), "stages": STAGES, "rows": rows,
           "summary": summary}
    with open("exp/result_minimal_information_model.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
