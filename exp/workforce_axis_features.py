"""H35: 研修対象人材・知識労働集約度の親軸。

購入ラベルを見ず、企業概要・組織図・業界・BtoB/BtoC・従業員数から次の4軸を
行単位で決定的に作る。

  1. 組織の知識職比率: 組織図に現れる知識職/現場職ファミリーの構成
  2. 事業の知識集約度: 企業概要の職種構成、業界の職務構造、BtoB度の単純平均
  3. 専門職の幅: 組織図と企業概要を合わせた知識職ファミリーのカバー率
  4. 推定研修対象規模: log(従業員数+1) × 上記2つの平均

語の生起回数は文章の長さや定型文に依存するため使わず、各職種ファミリーが1度でも
現れたかだけを使う。業界スコアは購入率で並べず、一般的な職務構成について事前固定した
ドメイン表である。欠損時は中立値0.5を返す。学習統計・target encodingは使わない。
"""
import re
import unicodedata

import numpy as np
import pandas as pd


WORKFORCE_AXIS_COLS = (
    "人材_組織知識職比率",
    "人材_事業知識集約度",
    "人材_専門職幅",
    "人材_推定研修対象規模",
)
WORKFORCE_INTERACTION_COLS = (
    "人材_log従業員",
    "人材_推定対象比率",
    "人材_推定研修対象規模",
)

# 広い単語（「営業」「管理」「開発」単独）は多くの会社に現れるため避ける。
KNOWLEDGE_FAMILIES = {
    "digital": r"DX|デジタル|\bIT\b|ICT|情報システム|システム開発|ソフトウェア|データ|AI|IOT|クラウド|セキュリティ",
    "research": r"研究|R&D|技術開発|商品開発|製品開発|イノベーション",
    "engineering": r"設計|エンジニア|技術(?:部|本部|センター)|品質保証|品質管理|知的財産|知財|解析|分析|試験|検査",
    "strategy": r"経営企画|事業企画|商品企画|製品企画|新規事業|戦略|マーケティング",
    "corporate_specialist": r"人事|財務|経理|法務|監査|コンプライアンス",
    "professional_service": r"コンサル|アドバイザ|専門職|士業|金融|保険|教育|研修|人材",
}

FIELD_FAMILIES = {
    "production": r"製造|生産|工場|加工|組立|設備保全",
    "construction": r"施工|建設|工事|土木|工務|現場作業",
    "logistics": r"物流|運輸|配送|倉庫|輸送|海運|鉄道|バス",
    "retail_hospitality": r"店舗|小売|接客|外食|飲食|ホテル|百貨店",
    "care_medical": r"介護|看護|診療|病院|福祉|保育",
    "primary_industry": r"農業|林業|漁業|鉱業|採掘",
}

_KNOWLEDGE_RE = {k: re.compile(v) for k, v in KNOWLEDGE_FAMILIES.items()}
_FIELD_RE = {k: re.compile(v) for k, v in FIELD_FAMILIES.items()}

# 購入率ではなく、企画・技術・専門職が業務の中心になりやすい度合いを事前に置いた表。
INDUSTRY_KNOWLEDGE_SCORE = {
    "IT": 1.00,
    "ゲーム": 0.95,
    "コンサルティング": 0.95,
    "専門サービス": 0.90,
    "金融": 0.90,
    "広告": 0.85,
    "人材": 0.85,
    "マスコミ": 0.80,
    "教育": 0.80,
    "通信": 0.80,
    "商社": 0.70,
    "エンタメ": 0.65,
    "不動産": 0.65,
    "機械関連サービス": 0.65,
    "通信機器": 0.65,
    "電気製品": 0.65,
    "機械": 0.60,
    "化学": 0.60,
    "自動車・乗り物": 0.50,
    "医療・福祉": 0.50,
    "エネルギー": 0.50,
    "その他サービス": 0.50,
    "その他": 0.50,
    "製造": 0.35,
    "生活用品": 0.35,
    "食品": 0.30,
    "アパレル・美容": 0.30,
    "建設・工事": 0.25,
    "小売": 0.20,
    "運輸・物流": 0.20,
    "外食": 0.15,
}


def _normalize(value):
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).upper()


def _family_hits(text, patterns):
    return {name for name, pattern in patterns.items() if pattern.search(text)}


def _composition(text):
    """知識職比率、知識職集合を返す。Laplace平滑化の中立点は0.5。"""
    normalized = _normalize(text)
    knowledge = _family_hits(normalized, _KNOWLEDGE_RE)
    field = _family_hits(normalized, _FIELD_RE)
    ratio = (len(knowledge) + 1.0) / (len(knowledge) + len(field) + 2.0)
    return ratio, knowledge


def _b2b_score(value):
    text = _normalize(value).replace(" ", "")
    has_b2b, has_b2c = "BTOB" in text, "BTOC" in text
    if has_b2b and not has_b2c:
        return 1.0
    if has_b2b and has_b2c:
        return 0.5
    if has_b2c:
        return 0.0
    return 0.5


def make_workforce_axes(raw):
    """H35の4軸だけを返す。入力行数とindexを維持する。"""
    rows = []
    for _, row in raw.iterrows():
        org_ratio, org_knowledge = _composition(row.get("組織図", ""))
        overview_ratio, overview_knowledge = _composition(row.get("企業概要", ""))
        industry = INDUSTRY_KNOWLEDGE_SCORE.get(row.get("業界"), 0.5)
        b2b = _b2b_score(row.get("特徴", ""))

        business = (overview_ratio + industry + b2b) / 3.0
        breadth = len(org_knowledge | overview_knowledge) / len(KNOWLEDGE_FAMILIES)
        target_share = (org_ratio + business) / 2.0
        employees = pd.to_numeric(row.get("従業員数"), errors="coerce")
        log_employees = np.log1p(max(float(employees), 0.0)) if pd.notna(employees) else 0.0
        rows.append((org_ratio, business, breadth, log_employees * target_share))
    return pd.DataFrame(rows, columns=WORKFORCE_AXIS_COLS, index=raw.index)


def add_workforce_axes(raw, X):
    out = X.copy()
    axes = make_workforce_axes(raw)
    for col in WORKFORCE_AXIS_COLS:
        out[col] = axes[col].to_numpy()
    return out


def make_workforce_interaction(raw):
    """H35b: 対象人口の親2本と積を返す。

    H35の事後分解で対象規模軸に信号が集中したための探索的追試。積の効果を強い片親と
    比べず、log従業員・推定対象比率という両親を同じLRへ必ず併記する。
    """
    axes = make_workforce_axes(raw)
    employees = pd.to_numeric(raw["従業員数"], errors="coerce").fillna(0).clip(lower=0)
    log_employees = np.log1p(employees.astype(float))
    scale = axes["人材_推定研修対象規模"]
    target_share = scale.div(log_employees.replace(0, np.nan)).fillna(0.5)
    return pd.DataFrame({
        "人材_log従業員": log_employees.to_numpy(),
        "人材_推定対象比率": target_share.to_numpy(),
        "人材_推定研修対象規模": scale.to_numpy(),
    }, index=raw.index)


def add_workforce_interaction(raw, X):
    out = X.copy()
    features = make_workforce_interaction(raw)
    for col in WORKFORCE_INTERACTION_COLS:
        out[col] = features[col].to_numpy()
    return out
