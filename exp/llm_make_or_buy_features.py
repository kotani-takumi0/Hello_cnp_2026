"""ラベル非依存で抽出したLLM Make-or-Buy特徴の読み込み。"""
import pandas as pd

PARENT_COLS = (
    "need_score", "make_limitation_score", "buy_openness_score",
    "internal_training_capability_score", "implementation_stage_ordinal",
    "evidence_strength", "need_unknown", "make_limitation_unknown",
    "buy_openness_unknown", "internal_training_capability_unknown",
    "implementation_stage_unknown",
)
PRODUCT_COLS = (
    "blocked_need", "need_buy", "limitation_buy", "make_or_buy_gap",
    "confidence_adjusted_gap", "internal_training_gap",
)

LLM_SOURCE_COLS = list(PARENT_COLS + PRODUCT_COLS) + ["llm_row_missing"]
LLM_FEATURE_COLS = [f"LLM_{c}" for c in LLM_SOURCE_COLS]


def add_llm_make_or_buy_features(raw, X, split):
    """企業IDで抽出結果を結合し、API失敗行を専用フラグ付きで保持する。"""
    path = f"data/llm_make_or_buy/{split}_make_or_buy_features.csv"
    feat = pd.read_csv(path)
    aligned = raw[["企業ID"]].merge(feat, on="企業ID", how="left",
                                     validate="one_to_one")
    parents = list(PARENT_COLS)
    aligned["llm_row_missing"] = aligned[parents].isna().all(axis=1).astype(int)
    values = aligned[LLM_SOURCE_COLS].fillna(-1).copy()
    values.columns = LLM_FEATURE_COLS
    out = X.copy()
    for col in LLM_FEATURE_COLS:
        out[col] = values[col].to_numpy()
    return out
