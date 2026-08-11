"""Organization chart features.

H13: `組織図` contains weak structure signal but useful lexical signal.
This module adds fixed row-wise keyword features and a nested TF-IDF+LR
stacking score, mirroring the H8 text stacking pattern.
"""
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ORG_COL = "組織図"
ORG_SCORE_COL = "組織図_購入確率"
INNER_SPLITS = 5
INNER_SEED = 0

ORG_TFIDF_PARAMS = dict(analyzer="char", ngram_range=(2, 6), min_df=3, sublinear_tf=True)
# random_state 必須。理由は text_features.LR_PARAMS のコメントを参照。
ORG_LR_PARAMS = dict(C=1.0, solver="liblinear", dual=True, max_iter=3000,
                     random_state=0)

_SPACE = re.compile(r"[\s　]+")


def _norm_series(s):
    return (
        s.fillna("")
        .astype(str)
        .map(lambda x: unicodedata.normalize("NFKC", x))
        .map(lambda x: _SPACE.sub(" ", x))
    )


def _has(text, patterns):
    pat = "|".join(re.escape(p) for p in patterns)
    return text.str.contains(pat, regex=True).astype(np.int8)


def add_org_manual_features(df, X):
    """Add fixed, leak-free organization chart features from the EDA report."""
    X = X.copy()
    text = _norm_series(df[ORG_COL])

    has_dx = _has(text, ("DX", "デジタルトランスフォーメーション"))
    has_dx_office = _has(text, ("DX推進室", "DX推進部", "DX推進本部", "DX推進"))
    has_tech = _has(text, ("技術", "技術開発", "研究開発", "開発"))
    has_planning = _has(text, ("経営企画", "戦略企画", "事業企画", "企画"))
    has_rd = _has(text, ("研究開発", "R&D", "研究所"))

    X["組織図_has_dx"] = has_dx
    X["組織図_has_dx推進"] = has_dx_office
    X["組織図_has_digital"] = _has(text, ("デジタル", "IT", "データ"))
    X["組織図_has_カード"] = _has(text, ("カード", "クレジット"))
    X["組織図_has_金融代理"] = _has(text, ("カード", "審査", "与信", "決済", "債権"))
    X["組織図_has_施工工事"] = _has(text, ("施工", "工事", "建設"))
    X["組織図_has_店舗"] = _has(text, ("店舗", "店長", "販売店"))
    X["組織図_has_不動産"] = _has(text, ("不動産", "賃貸", "物件"))
    X["組織図_has_技術開発"] = has_tech
    X["組織図_has_経営企画"] = has_planning
    X["組織図_has_研究開発"] = has_rd

    X["組織図_dx_x_技術開発"] = (has_dx * has_tech).astype(np.int8)
    X["組織図_dx_x_経営企画"] = (has_dx * has_planning).astype(np.int8)
    X["組織図_dx_x_研究開発"] = (has_dx * has_rd).astype(np.int8)
    return X


def load_org_text(df):
    """Return raw organization chart text.

    Raw text is intentional: the EDA found possible generation artifacts, which
    may be predictive under the competition's train/test generation process.
    """
    return df[ORG_COL].fillna("").astype(str).values


def build_org_model():
    return make_pipeline(
        TfidfVectorizer(**ORG_TFIDF_PARAMS),
        LogisticRegression(**ORG_LR_PARAMS),
    )


def fold_org_preds(txt_tr, y_tr, txt_va, txt_te, inner_seed=INNER_SEED):
    """Return (outer-train OOF, outer-valid pred, test pred) for org text."""
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=inner_seed)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_org_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_org_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1], full.predict_proba(txt_te)[:, 1]
