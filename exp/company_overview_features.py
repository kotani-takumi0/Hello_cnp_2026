"""Manual business-domain features from `企業概要`.

H16: deterministic features that use company overview as business-model and
digital-affinity signal, not as a direct supervised purchase-probability score.
"""
import re
import unicodedata

import numpy as np

OVERVIEW_COL = "企業概要"

_SPACE = re.compile(r"[\s　]+")

DIGITAL_PATTERNS = (
    "IT", "ICT", "DX", "デジタル", "システム", "ソフトウェア",
    "ソリューション", "クラウド", "データ", "IoT", "AI", "情報",
)
IT_SERVICE_PATTERNS = (
    "開発", "運用", "保守", "導入", "構築", "支援", "コンサル",
    "コンサルティング", "受託", "アウトソーシング",
)
RD_PATTERNS = ("研究開発", "R&D", "技術開発", "研究", "開発")
FINANCE_PATTERNS = (
    "金融", "銀行", "証券", "保険", "ローン", "信用", "決済",
    "クレジット", "投資", "カード",
)
MANUFACTURING_PATTERNS = (
    "製造", "メーカー", "工場", "部品", "機器", "装置", "半導体",
    "素材", "製品", "生産",
)
LOW_PURCHASE_PATTERNS = (
    "小売", "店舗", "百貨店", "スーパー", "販売店", "飲食", "料理",
    "食材", "建設", "施工", "工事", "土木", "不動産", "物流",
    "運輸", "鉄道", "ホテル", "レジャー", "医療", "福祉", "介護",
)
PHYSICAL_ASSET_PATTERNS = (
    "店舗", "施設", "工場", "倉庫", "物流", "運輸", "建設", "施工",
    "工事", "不動産", "ホテル", "鉄道", "設備",
)

TOP_HIGH_PURCHASE_GROUPS = (
    ("dx", ("DX", "デジタルトランスフォーメーション")),
    ("iot", ("IoT",)),
    ("it_ict", ("IT", "ICT")),
)
TOP_LOW_PURCHASE_GROUPS = (
    ("construction", ("建設", "施工", "工事", "土木")),
    ("education_training", ("教育", "研修", "人材育成", "リスキリング", "学習")),
    ("medical", ("医療", "病院", "医薬", "福祉", "介護")),
)


def _normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return _SPACE.sub(" ", text).strip()


def _count(text, patterns):
    return sum(text.count(p) for p in patterns)


def _has(texts, patterns):
    pat = "|".join(re.escape(p) for p in patterns)
    return texts.str.contains(pat, regex=True).astype(np.int8)


def add_company_overview_manual_features(df, X):
    """Add leak-free overview features from EDA findings."""
    X = X.copy()
    text = df[OVERVIEW_COL].fillna("").astype(str).map(_normalize)

    digital = text.map(lambda s: _count(s, DIGITAL_PATTERNS))
    service = text.map(lambda s: _count(s, IT_SERVICE_PATTERNS))
    rd = text.map(lambda s: _count(s, RD_PATTERNS))
    finance = text.map(lambda s: _count(s, FINANCE_PATTERNS))
    manufacturing = text.map(lambda s: _count(s, MANUFACTURING_PATTERNS))
    low = text.map(lambda s: _count(s, LOW_PURCHASE_PATTERNS))
    physical = text.map(lambda s: _count(s, PHYSICAL_ASSET_PATTERNS))

    X["概要_digital_count"] = digital.astype(np.int16)
    X["概要_it_service_count"] = service.astype(np.int16)
    X["概要_rd_count"] = rd.astype(np.int16)
    X["概要_finance_count"] = finance.astype(np.int16)
    X["概要_manufacturing_count"] = manufacturing.astype(np.int16)
    X["概要_low_purchase_domain_count"] = low.astype(np.int16)
    X["概要_physical_asset_count"] = physical.astype(np.int16)

    X["概要_has_digital"] = (digital > 0).astype(np.int8)
    X["概要_has_it_service"] = (service > 0).astype(np.int8)
    X["概要_has_rd"] = (rd > 0).astype(np.int8)
    X["概要_has_finance"] = (finance > 0).astype(np.int8)
    X["概要_has_low_purchase_domain"] = (low > 0).astype(np.int8)
    X["概要_has_physical_asset"] = (physical > 0).astype(np.int8)

    X["概要_digital_minus_low"] = (digital + service + rd + finance - low).astype(np.int16)
    X["概要_digital_x_service"] = ((digital > 0) & (service > 0)).astype(np.int8)
    X["概要_digital_x_rd"] = ((digital > 0) & (rd > 0)).astype(np.int8)
    X["概要_finance_x_digital"] = ((finance > 0) & (digital > 0)).astype(np.int8)
    X["概要_low_x_physical"] = ((low > 0) & (physical > 0)).astype(np.int8)

    if "業界" in df.columns:
        industry = df["業界"].fillna("").astype(str)
        is_information = industry.str.contains("情報|通信|IT|ソフト", regex=True)
        is_finance = industry.str.contains("金融|銀行|証券|保険", regex=True)
        is_manufacturing = industry.str.contains("製造|機械|電気|化学|素材", regex=True)
        X["概要_digital_x_non_info_industry"] = (
            (digital > 0) & (~is_information)
        ).astype(np.int8)
        X["概要_finance_x_non_finance_industry"] = (
            (finance > 0) & (~is_finance)
        ).astype(np.int8)
        X["概要_rd_x_manufacturing_industry"] = (
            (rd > 0) & is_manufacturing
        ).astype(np.int8)

    return X


def add_company_overview_top_flags(df, X, n_groups=3):
    """Add only the top high-purchase overview keyword flags from the EDA.

    Groups are cumulative in EDA order:
      1. DX
      2. IoT
      3. IT / ICT
    """
    X = X.copy()
    text = df[OVERVIEW_COL].fillna("").astype(str).map(_normalize)
    for name, patterns in TOP_HIGH_PURCHASE_GROUPS[:n_groups]:
        X[f"概要_top_{name}"] = _has(text, patterns)
    if n_groups > 1:
        cols = [f"概要_top_{name}" for name, _ in TOP_HIGH_PURCHASE_GROUPS[:n_groups]]
        X["概要_top_high_purchase_count"] = X[cols].sum(axis=1).astype(np.int8)
    return X


def add_company_overview_six_rate_flags(df, X):
    """Add top 3 high-rate and top 3 low-rate overview keyword flags."""
    X = X.copy()
    text = df[OVERVIEW_COL].fillna("").astype(str).map(_normalize)

    high_cols = []
    for name, patterns in TOP_HIGH_PURCHASE_GROUPS:
        col = f"概要_high_{name}"
        X[col] = _has(text, patterns)
        high_cols.append(col)

    low_cols = []
    for name, patterns in TOP_LOW_PURCHASE_GROUPS:
        col = f"概要_low_{name}"
        X[col] = _has(text, patterns)
        low_cols.append(col)

    X["概要_high3_count"] = X[high_cols].sum(axis=1).astype(np.int8)
    X["概要_low3_count"] = X[low_cols].sum(axis=1).astype(np.int8)
    X["概要_high3_minus_low3"] = (
        X["概要_high3_count"] - X["概要_low3_count"]
    ).astype(np.int8)
    return X
