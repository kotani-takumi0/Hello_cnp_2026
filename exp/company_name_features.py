"""`企業名` からの低自由度特徴（H20）。

EDA(`documents/company_name_eda_report.md`)の結論:
  - 企業名は全742行ユニーク → カテゴリ化/Target Encoding は無意味
  - char TF-IDF 単体は AUC 0.526（全件陽性のF1 0.3887 に対し最良F1 0.3982＝+0.0095）
  - `業界` に TF-IDF を足すと AUC 0.6541 -> 0.6276 と悪化。
    Bootstrap 95%CI [-0.0488, -0.0042] で全体が0より下 ＝ ノイズ追加の公算が高い
  - ただし文字構成には弱い有意差がある:
    カタカナ数 r=0.123 (p=0.0015) / 文字数 r=0.116 (p=0.0034) / 漢字数 r=-0.087 (p=0.0124)

そこでレポート14.4の方針に従い、**自由度の低い特徴を少数だけ**作る。
TF-IDF も n-gram フラグも作らない（FDR補正後に有意なn-gramは0個だった）。

エキスパートとして独立させず E6 の列として混ぜる前提。構成員を増やすと
742行では重み推定の自由度が増えて他の重みまで劣化することが実測されている
（E7 LLM: 重み4.0%しか取らないのに 4本構成に対し ΔAP -0.0009 で 0/5）。
"""
import re
import unicodedata

import numpy as np

NAME_COL = "企業名"

# 法人種別。文字数を測る前に落とす（EDAでも除去後に同じ差が残ることを確認済み）
_CORP_TYPES = (
    "株式会社", "有限会社", "合同会社", "合資会社", "合名会社",
    "一般社団法人", "一般財団法人", "公益社団法人", "公益財団法人",
)
_CORP_RE = re.compile("|".join(map(re.escape, _CORP_TYPES)))
_KATAKANA = re.compile(r"[ァ-ヶー]")
_KANJI = re.compile(r"[一-龥]")

# EDAで最も件数のある局所パターン（19件・購入率5.26%・Fisher p=0.0569）。
# 補正後は有意でないので、これ単独の採否も必ず複数seedで確認すること。
_GIKEN_RE = re.compile("技研")


def _normalize(text):
    return unicodedata.normalize("NFKC", str(text)).strip()


def _strip_corp(text):
    return _CORP_RE.sub("", text).strip()


def add_company_name_features(df, X, use_giken=True):
    """企業名から低自由度特徴のみを足す。行単位の決定的変換＝リーク無し。"""
    X = X.copy()
    name = df[NAME_COL].fillna("").astype(str).map(_normalize)
    core = name.map(_strip_corp)
    n = core.str.len().replace(0, np.nan)

    X["企業名_カタカナ比率"] = (
        core.map(lambda s: len(_KATAKANA.findall(s))) / n).fillna(0.0).astype(float)
    X["企業名_漢字比率"] = (
        core.map(lambda s: len(_KANJI.findall(s))) / n).fillna(0.0).astype(float)
    X["企業名_文字数"] = core.str.len().astype(np.int16)
    if use_giken:
        X["企業名_has_技研"] = core.str.contains(_GIKEN_RE).astype(np.int8)
    return X


VARIANTS = {
    "カタカナ比率のみ": ("企業名_カタカナ比率",),
    "漢字比率のみ": ("企業名_漢字比率",),
    "文字数のみ": ("企業名_文字数",),
    "技研フラグのみ": ("企業名_has_技研",),
    "比率2本": ("企業名_カタカナ比率", "企業名_漢字比率"),
    "全部(4本)": ("企業名_カタカナ比率", "企業名_漢字比率",
                "企業名_文字数", "企業名_has_技研"),
}
