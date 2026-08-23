"""現在のH2-both4構成でE2だけを購買状態特徴に差し替えて比較する。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import _scores, _survey_lr, blend_oof, build_features  # noqa: E402
from expert_groups import e2_survey_cols  # noqa: E402
from survey_purchase_state_features import (  # noqa: E402
    SURVEY_PURCHASE_STATE_COLS, add_survey_purchase_states,
)

SEED = 42
VARIANTS = {
    "intent_execution": [SURVEY_PURCHASE_STATE_COLS[0]],
    "adopted_dissatisfied": [SURVEY_PURCHASE_STATE_COLS[1]],
    "problem_external": [SURVEY_PURCHASE_STATE_COLS[2]],
    "lowmake_highbuy": [SURVEY_PURCHASE_STATE_COLS[3]],
    "states4": list(SURVEY_PURCHASE_STATE_COLS),
}


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X, Xte = add_survey_purchase_states(train, X), add_survey_purchase_states(test, Xte)
    path = "exp/_experts_seed42_concatemb_crosslr_lin0.03_llmcore4e0onlyh2both4.npz"
    d = np.load(path)
    base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
    base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
    base_pred, _, _, _ = blend_oof(y, base_oof, base_te, SEED, experts=BASE_EXPERTS)
    base_score = _scores(y, base_pred)
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base_cols = e2_survey_cols()
    for name, extra in VARIANTS.items():
        cols = base_cols + extra
        oof, tests = np.zeros(len(X)), []
        for tr, va in cv.split(X, y):
            p, q = _survey_lr(X.iloc[tr][cols], y[tr], X.iloc[va][cols], Xte[cols])
            oof[va] = p
            tests.append(q)
        oo, tt = dict(base_oof), dict(base_te)
        oo["E2_survey"] = oof
        tt["E2_survey"] = np.mean(tests, axis=0)
        pred, pred_te, _, _ = blend_oof(y, oo, tt, SEED, experts=BASE_EXPERTS)
        score = _scores(y, pred)
        row = {"variant": name, "e2": _scores(y, oof), "final": score,
               "delta": {m: float(score[m] - base_score[m]) for m in ("auc", "ap", "f1")}}
        rows.append(row)
        print(name, row, flush=True)
    with open("exp/result_e2_states_current.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": base_score, "rows": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
