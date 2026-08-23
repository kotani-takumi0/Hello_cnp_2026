"""pruned E0へH47〜H51の代表情報だけを戻す。E0bは元の94列で固定。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buying_center_features import add_buying_center_features  # noqa: E402
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from distributed_rollout_features import add_distributed_rollout_features  # noqa: E402
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, _lgbm, _nested_text, _scores, blend_oof, build_features,
)
from expert_groups import FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols  # noqa: E402
from llm_make_or_buy_features import add_llm_make_or_buy_features  # noqa: E402
from make_or_buy_features import add_make_or_buy_features  # noqa: E402
from organization_features import ORG_SCORE_COL, fold_org_preds, load_org_text  # noqa: E402
from procurement_timing_features import add_procurement_timing_features  # noqa: E402
from readiness_to_scale_features import add_readiness_features  # noqa: E402
from text_features import TEXT_COL  # noqa: E402

SEEDS = (0, 1, 2, 3, 42)
CORE4 = ["LLM_need_score", "LLM_make_limitation_score",
         "LLM_buy_openness_score", "LLM_internal_training_capability_score"]
RESTORES = {
    "H47_buying_center": ["購買主体_完成度", "購買主体_三者連携"],
    "H48_make_buy": ["MakeBuy_外部minus内部", "MakeBuy_外部支援明示"],
    "H49_stage": ["展開段階_移行幅", "展開段階_scale_window"],
    "H49_transition_only": ["展開段階_移行幅"],
    "H49_scale_window_only": ["展開段階_scale_window"],
    "H50_rollout_load": ["展開負荷_拠点x全社展開", "展開負荷_拠点x教育展開"],
    "H51_timing": ["購買時点_具体性", "購買時点_実行準備完了"],
}


def main():
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test)
    X = add_llm_make_or_buy_features(train, X, "train")
    Xte = add_llm_make_or_buy_features(test, Xte, "test")
    for add in (add_buying_center_features, add_make_or_buy_features,
                add_readiness_features, add_distributed_rollout_features,
                add_procurement_timing_features):
        X, Xte = add(train, X), add(test, Xte)
    original = e0_anchor_cols(build_features(train, test)[0].columns)
    missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
    retained = [c for c in original if not c.startswith("DX展望_") and
                c not in missing]
    txt = train[TEXT_COL].fillna("").astype(str).values
    txt_te = test[TEXT_COL].fillna("").astype(str).values
    org, org_te = load_org_text(train), load_org_text(test)
    rows = []
    for seed in SEEDS:
        d = np.load(f"exp/_experts_seed{seed}_concatemb_crosslr_lin0.03_llmcore4e0only.npz")
        base_oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
        base_te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
        base, _, _, _ = blend_oof(y, base_oof, base_te, seed,
                                  experts=BASE_EXPERTS)
        sb = _scores(y, base)
        po = {v: np.zeros(len(X)) for v in RESTORES}
        pt = {v: [] for v in RESTORES}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, va in cv.split(X, y):
            ttr, tva, tte = _nested_text(txt[tr], y[tr], txt[va], txt_te)
            otr, ova, ote = fold_org_preds(org[tr], y[tr], org[va], org_te)
            for variant, extra in RESTORES.items():
                cols = retained + CORE4 + extra
                Atr, Ava, Ate = X.iloc[tr][cols].copy(), X.iloc[va][cols].copy(), Xte[cols].copy()
                for col, vals in ((DX_SCORE_COL, (ttr, tva, tte)),
                                  (ORG_SCORE_COL, (otr, ova, ote))):
                    Atr[col], Ava[col], Ate[col] = vals
                p, q = _lgbm(Atr, y[tr], Ava, y[va], Ate, seed)
                po[variant][va], _ = p, q
                pt[variant].append(q)
        for variant in RESTORES:
            oof, te = dict(base_oof), dict(base_te)
            oof["E0_anchor"] = po[variant]
            te["E0_anchor"] = np.mean(pt[variant], axis=0)
            pred, _, _, _ = blend_oof(y, oof, te, seed, experts=BASE_EXPERTS)
            s = _scores(y, pred)
            delta = {k: float(s[k] - sb[k]) for k in ("auc", "ap", "f1")}
            rows.append({"seed": seed, "variant": variant, "delta": delta})
            print(seed, variant, delta)
    summary = {}
    for variant, cols in RESTORES.items():
        rs = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"columns": cols}
        for metric in ("auc", "ap", "f1"):
            vals = np.array([r["delta"][metric] for r in rs])
            summary[variant][metric] = {"mean": float(vals.mean()),
                                        "wins": int((vals > 0).sum()), "n": 5}
    with open("exp/result_pruned_restore_h47_h51.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
