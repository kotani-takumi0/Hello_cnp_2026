"""データ読み込みと特徴量行列の組み立て（exp004〜007 と同一手順）。

`make_submission_h1_h8.py` が持っていた preprocess / add_h1 / join_text を、
アンサンブル実験から再利用できるように切り出したもの。処理内容は同一
（exp004〜007 の再現性を壊さないため、既存スクリプト側は変更していない）。

テキスト列はここでは特徴にしない。fold 内で完結させる必要があるので
生テキストだけ返し、fold 分割後に `text_features` 側で確率1本に潰す。
"""
import numpy as np
import pandas as pd

from llm_features import add_llm_axes
from dx_outlook_features import add_dx_outlook_manual_features
from company_overview_features import add_company_overview_manual_features
from organization_features import add_org_manual_features, load_org_text

TARGET = "購入フラグ"
ID_COL = "企業ID"
DROP_TEXT = ["企業ID", "企業名", "企業概要", "組織図", "今後のDX展望"]
CAT_COLS = ["業界", "上場種別", "特徴"]
ZERO_FILL = ["事業所数", "工場数", "店舗数", "アンケート７"]
FLAG_ONLY = ["資本金", "営業利益", "経常利益"]
TEXT_COL = "今後のDX展望"
OVERVIEW_COL = "企業概要"


def _safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def add_h1(df, X):
    """H1: 規模で正規化した利益率3本。"""
    X = X.copy()
    X["営業利益率"] = _safe_div(df["営業利益"], df["売上"])
    X["経常利益率"] = _safe_div(df["経常利益"], df["売上"])
    X["純利益率"] = _safe_div(df["当期純利益"], df["売上"])
    return X


def preprocess(df, cat_categories=None):
    """テキスト列を落とし、欠損フラグを立て、カテゴリ dtype を固定する。"""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_TEXT if c in df.columns])
    for c in ZERO_FILL:
        df[f"{c}_isna"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0)
    for c in FLAG_ONLY:
        df[f"{c}_isna"] = df[c].isna().astype(int)
    for c in CAT_COLS:
        if cat_categories is None:
            df[c] = df[c].astype("category")
        else:
            df[c] = pd.Categorical(df[c], categories=cat_categories[c])
    return df


def join_text(df, cols):
    """テキスト列を空白で結合して ndarray で返す。"""
    s = df[cols[0]].fillna("")
    for c in cols[1:]:
        s = s + " " + df[c].fillna("")
    return s.values


def load_frames(data_dir="data"):
    return (pd.read_csv(f"{data_dir}/train.csv"),
            pd.read_csv(f"{data_dir}/test.csv"))


def build_matrices(train, test, use_llm=False, use_org_chart=False,
                   use_dx_outlook_manual=False,
                   use_company_overview_manual=False, data_dir="data"):
    """(X, y, Xte) を返す。train のカテゴリ定義を test にも適用して列を揃える。"""
    tp = preprocess(train)
    cat_categories = {c: tp[c].cat.categories for c in CAT_COLS}
    y = tp[TARGET].values.astype(int)
    X = add_h1(train, tp.drop(columns=[TARGET]))

    tp_test = preprocess(test, cat_categories).drop(columns=[TARGET], errors="ignore")
    Xte = add_h1(test, tp_test)

    if use_org_chart:
        X = add_org_manual_features(train, X)
        Xte = add_org_manual_features(test, Xte)

    if use_dx_outlook_manual:
        X = add_dx_outlook_manual_features(train, X)
        Xte = add_dx_outlook_manual_features(test, Xte)

    if use_company_overview_manual:
        X = add_company_overview_manual_features(train, X)
        Xte = add_company_overview_manual_features(test, Xte)

    if use_llm:
        X = add_llm_axes(train, X, f"{data_dir}/train_with_llm_3axes.csv")
        Xte = add_llm_axes(test, Xte, f"{data_dir}/test_with_llm_3axes.csv")

    Xte = Xte[X.columns]
    assert list(X.columns) == list(Xte.columns), "train/test の列順が不一致"
    return X, y, Xte


def build_texts(train, test, use_overview=False):
    """(train テキスト, test テキスト, 生成される特徴名) を返す。"""
    cols = (OVERVIEW_COL, TEXT_COL) if use_overview else (TEXT_COL,)
    name = "テキスト_購入確率" if use_overview else "DX展望_購入確率"
    return join_text(train, cols), join_text(test, cols), name


def build_org_texts(train, test):
    """(train 組織図テキスト, test 組織図テキスト) を返す。"""
    return load_org_text(train), load_org_text(test)
