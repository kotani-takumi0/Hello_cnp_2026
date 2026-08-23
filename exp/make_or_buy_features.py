"""H48: DX教育商材のMake-or-Buyギャップを表す固定特徴。

仮説は「DX能力が高いほど買う」ではない。外部の知識・支援を調達する姿勢が強い一方、
社内だけでDXを実行・展開する能力が不足する企業ほど、外部教育商材を購入しやすい。

購入ラベル、train内統計、分位点は使わない。アンケートの既知の1〜5尺度を0〜1へ写像し、
組織図とDX展望は評価前に固定した辞書で行単位変換する。
"""
import re
import unicodedata

import numpy as np
import pandas as pd


MAKE_OR_BUY_COLS = (
    "MakeBuy_外部調達性",
    "MakeBuy_内部実行力",
    "MakeBuy_外部minus内部",
    "MakeBuy_外部x能力不足",
    "MakeBuy_高外部低内部",
    "MakeBuy_高外部高内部",
    "MakeBuy_外部支援明示",
    "MakeBuy_内部組織能力",
)

_SPACE = re.compile(r"[\s　]+")
VENDOR_PATTERNS = (
    "外部", "ベンダー", "パートナー", "コンサル", "専門家", "外注", "委託",
    "伴走", "支援会社", "外部研修", "外部講師",
)
DX_PATTERNS = ("DX推進", "DX戦略", "デジタル推進", "デジタル戦略", "デジタル変革")
IT_PATTERNS = ("情報システム", "システム企画", "IT戦略", "IT企画", "IT推進")
DATA_PATTERNS = ("データ戦略", "データ分析", "データ活用", "AI推進", "データサイエンス")
DEVELOPMENT_PATTERNS = ("研究開発", "R&D", "技術開発", "システム開発", "開発本部")


def _normalize(value):
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    return _SPACE.sub(" ", text).strip()


def _has(text, patterns):
    return float(any(pattern in text for pattern in patterns))


def _survey01(raw, columns, weights=None):
    values = raw[list(columns)].apply(pd.to_numeric, errors="coerce")
    # 配布仕様の1〜5を固定端点で0〜1化。欠損時はその行の観測項目だけで平均する。
    values = ((values - 1.0) / 4.0).clip(0.0, 1.0)
    if weights is None:
        return values.mean(axis=1).fillna(0.5)
    weighted = values.mul(np.asarray(weights), axis=1)
    denominator = values.notna().mul(np.asarray(weights), axis=1).sum(axis=1)
    return weighted.sum(axis=1).div(denominator.replace(0, np.nan)).fillna(0.5)


def make_make_or_buy_features(raw):
    # Q10は外部連携実績そのものなので、セミナー参加(Q9)・情報収集(Q11)の2倍重視。
    external_survey = _survey01(
        raw, ("アンケート９", "アンケート１０", "アンケート１１"), (1, 2, 1),
    )
    # 導入(Q3)、セキュリティ基盤(Q5)、成果実感(Q8)を社内実行能力の代理とする。
    internal_survey = _survey01(raw, ("アンケート３", "アンケート５", "アンケート８"))

    outlook = raw["今後のDX展望"].map(_normalize)
    vendor = outlook.map(lambda text: _has(text, VENDOR_PATTERNS))
    organization = raw["組織図"].map(_normalize)
    org_capacity = organization.map(lambda text: np.mean([
        _has(text, DX_PATTERNS), _has(text, IT_PATTERNS),
        _has(text, DATA_PATTERNS), _has(text, DEVELOPMENT_PATTERNS),
    ]))

    # 連続親軸を残し、差と「需要×不足」を別々に渡す。LRが符号を学べるようにする。
    external = 0.75 * external_survey + 0.25 * vendor
    internal = 0.75 * internal_survey + 0.25 * org_capacity
    gap = external - internal
    unmet = external * (1.0 - internal)
    # 3.5/5以上を高外部、3/5以下を低内部とする固定された業務上の境界。
    high_external = external >= 0.625
    low_internal = internal <= 0.5

    return pd.DataFrame({
        MAKE_OR_BUY_COLS[0]: external,
        MAKE_OR_BUY_COLS[1]: internal,
        MAKE_OR_BUY_COLS[2]: gap,
        MAKE_OR_BUY_COLS[3]: unmet,
        MAKE_OR_BUY_COLS[4]: (high_external & low_internal).astype(np.int8),
        MAKE_OR_BUY_COLS[5]: (high_external & ~low_internal).astype(np.int8),
        MAKE_OR_BUY_COLS[6]: vendor.astype(np.int8),
        MAKE_OR_BUY_COLS[7]: org_capacity,
    }, index=raw.index)


def add_make_or_buy_features(raw, X):
    out = X.copy()
    features = make_make_or_buy_features(raw)
    for col in MAKE_OR_BUY_COLS:
        out[col] = features[col].to_numpy()
    return out
