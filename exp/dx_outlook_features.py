"""Manual features for `今後のDX展望`.

H14: Deterministic row-wise features from the DX outlook EDA.
The text stacking feature (H8) already captures broad lexical signal, while
these features expose interpretable axes that trees can combine with finance,
survey, and organization features.
"""
import re
import unicodedata

import numpy as np

TEXT_COL = "今後のDX展望"

_NUM = re.compile(r"[0-9]+(?:[,.][0-9]+)*")
_SPACE = re.compile(r"[\s　]+")
_SENT = re.compile(r"(?<=。)")
_KATAKANA = re.compile(r"[ァ-ヶー]")

CURRENT_MARK = (
    "これまで", "従来", "現状", "既存", "現在", "これまでは", "に過ぎ",
    "とどま", "でした", "ました", "きました", "残っており", "が実情",
    "限定的", "不十分",
)
FUTURE_MARK = (
    "今後", "将来", "将来的", "これから", "次期", "今後は", "今後の方針",
    "計画", "方針", "予定", "検討", "していく", "進める", "図る",
    "目指", "する考え", "拡大し", "整備し", "強化", "推進",
)

CAUTIOUS_PATTERNS = (
    "慎重", "必要最小限", "最小限", "必要最低限", "最低限", "限定",
    "小規模", "見送", "抑え", "抑制", "維持", "当面", "とどめ",
    "段階的", "必要に応じ", "優先順位", "大規模投資", "追加投資",
)
EXPAND_PATTERNS = (
    "加速", "拡大", "強化", "全社展開", "高度化", "深化", "実装",
    "積極投資", "次世代", "競争優位", "意思決定", "データドリブン",
    "クラウド", "AI", "人工知能", "自動化", "デジタル基盤",
)
EDUCATION_PATTERNS = (
    "教育", "研修", "人材育成", "人材開発", "リスキリング", "eラーニング",
    "Eラーニング", "社内大学", "アカデミー", "資格取得", "学習",
    "スキル", "DX人材", "デジタル人材", "育成",
)
VENDOR_PATTERNS = (
    "外部", "ベンダー", "パートナー", "コンサル", "支援", "専門家",
    "外注", "委託", "サービス",
)


def _normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = _NUM.sub("0", text)
    text = _SPACE.sub(" ", text)
    return text.strip()


def _split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _contains_any(text, patterns):
    return any(p in text for p in patterns)


def _count_patterns(text, patterns):
    return sum(text.count(p) for p in patterns)


def _future_text(text):
    sentences = _split_sentences(text)
    future = []
    for sent in sentences:
        has_future = _contains_any(sent, FUTURE_MARK)
        has_current = _contains_any(sent, CURRENT_MARK)
        if has_future or (not has_current and future):
            future.append(sent)
    return " ".join(future)


def _education_context(text):
    sentences = _split_sentences(text)
    return " ".join(s for s in sentences if _contains_any(s, EDUCATION_PATTERNS))


def _safe_ratio(num, den):
    return num / den if den else 0.0


def add_dx_outlook_manual_features(df, X):
    """Add leak-free dictionary and context features from `今後のDX展望`."""
    X = X.copy()
    texts = df[TEXT_COL].fillna("").astype(str).map(_normalize)
    future = texts.map(_future_text)
    education = texts.map(_education_context)

    text_len = texts.map(len).astype(float)
    future_len = future.map(len).astype(float)
    edu_len = education.map(len).astype(float)

    cautious = texts.map(lambda s: _count_patterns(s, CAUTIOUS_PATTERNS))
    expand = texts.map(lambda s: _count_patterns(s, EXPAND_PATTERNS))
    education_count = texts.map(lambda s: _count_patterns(s, EDUCATION_PATTERNS))
    vendor_count = texts.map(lambda s: _count_patterns(s, VENDOR_PATTERNS))
    future_cautious = future.map(lambda s: _count_patterns(s, CAUTIOUS_PATTERNS))
    future_expand = future.map(lambda s: _count_patterns(s, EXPAND_PATTERNS))
    future_education = future.map(lambda s: _count_patterns(s, EDUCATION_PATTERNS))
    edu_cautious = education.map(lambda s: _count_patterns(s, CAUTIOUS_PATTERNS))
    edu_expand = education.map(lambda s: _count_patterns(s, EXPAND_PATTERNS))

    X["DX展望_cautious_count"] = cautious.astype(np.int16)
    X["DX展望_expand_count"] = expand.astype(np.int16)
    X["DX展望_education_count"] = education_count.astype(np.int16)
    X["DX展望_vendor_count"] = vendor_count.astype(np.int16)
    X["DX展望_future_cautious_count"] = future_cautious.astype(np.int16)
    X["DX展望_future_expand_count"] = future_expand.astype(np.int16)
    X["DX展望_future_education_count"] = future_education.astype(np.int16)
    X["DX展望_edu_cautious_count"] = edu_cautious.astype(np.int16)
    X["DX展望_edu_expand_count"] = edu_expand.astype(np.int16)

    X["DX展望_has_cautious"] = (cautious > 0).astype(np.int8)
    X["DX展望_has_future_cautious"] = (future_cautious > 0).astype(np.int8)
    X["DX展望_has_education"] = (education_count > 0).astype(np.int8)
    X["DX展望_has_future_education"] = (future_education > 0).astype(np.int8)
    X["DX展望_has_vendor"] = (vendor_count > 0).astype(np.int8)
    X["DX展望_expand_minus_cautious"] = (expand - cautious).astype(np.int16)
    X["DX展望_future_expand_minus_cautious"] = (
        future_expand - future_cautious
    ).astype(np.int16)

    X["DX展望_future_text_ratio"] = [
        _safe_ratio(a, b) for a, b in zip(future_len, text_len)
    ]
    X["DX展望_education_text_ratio"] = [
        _safe_ratio(a, b) for a, b in zip(edu_len, text_len)
    ]
    X["DX展望_katakana_ratio"] = [
        _safe_ratio(len(_KATAKANA.findall(s)), len(s)) for s in texts
    ]

    X["DX展望_education_x_expand"] = (
        (education_count > 0) & (expand > 0)
    ).astype(np.int8)
    X["DX展望_education_x_cautious"] = (
        (education_count > 0) & (cautious > 0)
    ).astype(np.int8)
    X["DX展望_vendor_x_education"] = (
        (vendor_count > 0) & (education_count > 0)
    ).astype(np.int8)
    return X
