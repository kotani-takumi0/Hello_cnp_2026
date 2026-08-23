"""H50: 多拠点へのDX教育展開負荷を表す固定特徴。

仮説: 拠点・工場・店舗に人員が分散し、かつDXの全社展開や教育展開を予定する企業では、
標準化された外部教育商材の価値が高い。購入ラベルやtrain内統計は使わない。
"""
import re
import unicodedata

import numpy as np
import pandas as pd


DISTRIBUTED_ROLLOUT_COLS = (
    "展開負荷_log拠点数",
    "展開負荷_拠点種類数",
    "展開負荷_現場拠点あり",
    "展開負荷_log従業員",
    "展開負荷_全社横展開意図",
    "展開負荷_教育展開意図",
    "展開負荷_拠点x全社展開",
    "展開負荷_拠点x教育展開",
)

_SENTENCE = re.compile(r"(?<=[。！？])|[\r\n]+")
_SPACE = re.compile(r"[\s　]+")
SCALE_PATTERNS = (
    "全社展開", "全社的に展開", "全社横断", "全社一体", "全社規模",
    "全部門", "部門横断", "横展開", "水平展開", "全拠点", "各拠点",
    "グループ全体", "ロールアウト", "標準化",
)
EDUCATION_PATTERNS = (
    "教育", "研修", "人材育成", "人材開発", "リスキリング", "学習機会",
    "DX人材", "デジタル人材", "スキル向上", "リテラシー向上",
)
FUTURE_PATTERNS = (
    "今後", "これから", "予定", "方針", "計画", "目指", "していく",
    "まいります", "進める", "拡大", "強化",
)
BLOCK_PATTERNS = (
    "見送", "凍結", "中止", "行わない", "実施しない", "至っていない",
    "踏み込まない", "予定はない", "限定", "とどめ", "抑制",
)


def _normalize(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text).strip()


def _intent(text, target_patterns):
    """未来文脈かつ否定・見送りでない文に対象語があれば1。"""
    for sentence in (part.strip() for part in _SENTENCE.split(text)):
        if not sentence or any(block in sentence for block in BLOCK_PATTERNS):
            continue
        if (any(target in sentence for target in target_patterns)
                and any(future in sentence for future in FUTURE_PATTERNS)):
            return 1.0
    return 0.0


def make_distributed_rollout_features(raw):
    site_frame = raw[["事業所数", "工場数", "店舗数"]].apply(
        pd.to_numeric, errors="coerce",
    ).fillna(0).clip(lower=0)
    sites = site_frame.sum(axis=1)
    site_types = (site_frame > 0).sum(axis=1)
    field_sites = ((site_frame["工場数"] > 0) | (site_frame["店舗数"] > 0)).astype(float)
    employees = pd.to_numeric(raw["従業員数"], errors="coerce").fillna(0).clip(lower=0)
    texts = raw["今後のDX展望"].map(_normalize)
    scale_intent = texts.map(lambda text: _intent(text, SCALE_PATTERNS))
    education_intent = texts.map(lambda text: _intent(text, EDUCATION_PATTERNS))
    log_sites = np.log1p(sites)

    return pd.DataFrame({
        DISTRIBUTED_ROLLOUT_COLS[0]: log_sites,
        DISTRIBUTED_ROLLOUT_COLS[1]: site_types.astype(float),
        DISTRIBUTED_ROLLOUT_COLS[2]: field_sites,
        DISTRIBUTED_ROLLOUT_COLS[3]: np.log1p(employees),
        DISTRIBUTED_ROLLOUT_COLS[4]: scale_intent,
        DISTRIBUTED_ROLLOUT_COLS[5]: education_intent,
        DISTRIBUTED_ROLLOUT_COLS[6]: log_sites * scale_intent,
        DISTRIBUTED_ROLLOUT_COLS[7]: log_sites * education_intent,
    }, index=raw.index)


def add_distributed_rollout_features(raw, X):
    out = X.copy()
    features = make_distributed_rollout_features(raw)
    for col in DISTRIBUTED_ROLLOUT_COLS:
        out[col] = features[col].to_numpy()
    return out
