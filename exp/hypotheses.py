"""仮説レジストリ。
1つの仮説 = それが主張する概念を表現する特徴量群（1本とは限らない）。
各関数は (train_raw, X) -> X で、行単位の決定的変換のみ行う（＝リーク無し・安全）。
target encoding 等の fold依存な仮説は別枠(NEEDS_CV)で扱う。
"""
import numpy as np

from dx_outlook_features import add_dx_outlook_manual_features
from company_overview_features import add_company_overview_manual_features
from llm_features import add_llm_axes

CAT_COLS = ["業界", "上場種別", "特徴"]  # target encoding対象（harnessと一致）


def _safe_div(a, b):
    r = a / b
    return r.replace([np.inf, -np.inf], np.nan)


def h1_profit_margins(train, X):
    """H1 [A1 収益性]: 規模で正規化した利益率。ratio が生の利益額より効くか。"""
    X = X.copy()
    X["営業利益率"] = _safe_div(train["営業利益"], train["売上"])
    X["経常利益率"] = _safe_div(train["経常利益"], train["売上"])
    X["純利益率"] = _safe_div(train["当期純利益"], train["売上"])
    return X


def h2_financial_health(train, X):
    """H2 [C3 財務健全性]: 投資余力の代理。健全な企業ほど裁量的投資できる。"""
    X = X.copy()
    X["自己資本比率"] = _safe_div(train["自己資本"], train["総資産"])
    X["負債比率"] = _safe_div(train["負債"], train["自己資本"])
    X["ROA"] = _safe_div(train["当期純利益"], train["総資産"])
    X["ROE"] = _safe_div(train["当期純利益"], train["自己資本"])
    return X


def h3_digital_density(train, X):
    """H3 [C2 デジタル密度/1人あたり]: 従業員あたりのソフト投資・生産性。"""
    X = X.copy()
    emp = train["従業員数"].replace(0, np.nan)
    X["ソフト投資_per従業員"] = _safe_div(train["無形固定資産変動(ソフトウェア関連)"], emp)
    X["売上_per従業員"] = _safe_div(train["売上"], emp)
    X["総資産_per従業員"] = _safe_div(train["総資産"], emp)
    return X


def h4_need_gap(train, X):
    """H4 [A7 ニーズギャップ]: 現状不満・課題感の合成。満足度が低いほど購買駆動。"""
    X = X.copy()
    # 満足度系（高いほど満足）: 低いほど買う → 反転して不満スコア化
    sat = train[["アンケート２", "アンケート８"]].mean(axis=1)
    X["不満スコア"] = 6 - sat  # 1..5反転
    # 戦略明確(1)と満足(2)のギャップ: やりたいのに満足してない = 課題感
    X["戦略_満足ギャップ"] = train["アンケート１"] - train["アンケート２"]
    # 既存ツール満足度(7)は48%構造欠損。導入済み内での不満のみ効く → fill済み列を反転
    X["既存ツール不満"] = np.where(train["アンケート７"].isna(), 0, 6 - train["アンケート７"].fillna(0))
    return X


def h4b_need_gap_v2(train, X):
    """H4b [A7 ニーズギャップ・忠実版]: 意欲−実現のDXギャップ。
    H4の反省: 雑な合成が概念を裏切った(na と満足を混同)。今回は軸を明示。
    - 意欲軸 = 戦略明確(Q1)+情報収集(Q11)+セミナー参加(Q9)  … やりたい度
    - 実現軸 = 技術導入(Q3)+成果実感(Q8)+満足(Q2)          … 達成度
    - DXギャップ = 意欲 − 実現 (高い＝motivated but stuck＝本命)。木が作れない6項目差。
    - ツール不満: 導入済み(Q6=1)のみ (5−Q7)。未導入/満足は0。na と混ぜない。
    """
    X = X.copy()
    intent = train[["アンケート１", "アンケート１１", "アンケート９"]].mean(axis=1)
    achieve = train[["アンケート３", "アンケート８", "アンケート２"]].mean(axis=1)
    X["DX意欲"] = intent
    X["DX実現"] = achieve
    X["DXギャップ"] = intent - achieve
    is_adopter = train["アンケート６"] == 1
    X["ツール不満"] = np.where(is_adopter, 5 - train["アンケート７"].fillna(5), 0.0)
    return X


def h7_cash_slack(train, X):
    """H7 [② A1別軸 現金ベースの投資余力]: 教材費は会計利益でなく現金から出る。
    H1(会計利益率)と直交。連続財務どうしの比＝boostingも再構築しにくい"効くタイプ"。
    - 営業CFマージン = 営業CF / 売上         … 稼ぎの現金化度
    - FCF比率        = (営業CF+投資CF)/総資産 … 投資後に残る自由資金(投資CFは負)
    - 現金ROA        = 営業CF / 総資産        … 資産あたり現金創出力
    """
    X = X.copy()
    X["営業CFマージン"] = _safe_div(train["営業CF"], train["売上"])
    X["FCF比率"] = _safe_div(train["営業CF"] + train["投資CF"], train["総資産"])
    X["現金ROA"] = _safe_div(train["営業CF"], train["総資産"])
    return X


def h9_llm_axes(train, X):
    """H9 [テキスト由来・LLM評価]: `今後のDX展望` をLLMに読ませた3軸スコア。
    H8(TF-IDF char n-gram)と出所は同じテキストだが、圧縮の仕方が違う。
    H8は語彙の統計を教師ありで1次元に潰す。H9は意味を人間が指定した3軸
    （DX実行力/外部ベンダー必要度/慎重さ）に落とす。
    LLMは正解を見ていないので行単位の決定的変換＝リーク無し。
    """
    return add_llm_axes(train, X)


def h14_dx_outlook_manual(train, X):
    """H14 [DX展望の明示前処理]: 慎重/拡大/教育/未来文脈の辞書特徴。

    H8の教師ありTF-IDF確率は維持しつつ、EDAで強かった「慎重・抑制」軸と
    DX教材に直結する「教育・研修」文脈を、他特徴量と交互作用しやすい数値にする。
    行単位の決定的変換なのでリーク無し。
    """
    return add_dx_outlook_manual_features(train, X)


def h16_company_overview_manual(train, X):
    """H16 [企業概要の明示特徴]: 業態/IT親和性/低購入業態の辞書特徴。"""
    return add_company_overview_manual_features(train, X)


# 行単位で安全に足せる仮説（リーク無し）
REGISTRY = {
    "H1": ("収益性の正規化(利益率)", h1_profit_margins),
    "H2": ("財務健全性", h2_financial_health),
    "H3": ("デジタル密度/1人あたり", h3_digital_density),
    "H4": ("ニーズギャップ(不満)", h4_need_gap),
    "H4b": ("ニーズギャップ忠実版(DXギャップ)", h4b_need_gap_v2),
    "H7": ("現金ベースの投資余力(CF比率)", h7_cash_slack),
    "H9": ("LLM3軸(DX実行力/外部ベンダー必要度/慎重さ)", h9_llm_axes),
    "H14": ("DX展望の明示前処理(慎重/拡大/教育/未来文脈)", h14_dx_outlook_manual),
    "H16": ("企業概要の明示特徴(業態/IT親和性/低購入業態)", h16_company_overview_manual),
}

def _smooth_te(train_col, y_tr, val_col, m=10.0):
    """平滑化target encoding。train統計のみでfit。未知カテゴリはglobal。"""
    g = y_tr.mean()
    s = train_col.astype(object)
    stats = y_tr.groupby(s.values).agg(["mean", "count"])
    enc = (stats["count"] * stats["mean"] + m * g) / (stats["count"] + m)
    return val_col.astype(object).map(enc).fillna(g).astype(float).values


def h5_industry_te(Xtr, ytr, Xva):
    """H5 [A4/B3 業界エンコード]: 業界/上場種別/特徴 の target encoding。
    少数カテゴリの過学習を平滑化で抑えつつ購入率シグナルを数値化。"""
    import pandas as pd
    ytr = pd.Series(ytr, index=Xtr.index)
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for c in CAT_COLS:
        Xtr[f"{c}_te"] = _smooth_te(Xtr[c], ytr, Xtr[c])
        Xva[f"{c}_te"] = _smooth_te(Xtr[c], ytr, Xva[c])
    return Xtr, Xva


# fold内でfitが必要な仮説（target encoding等）。fold_transform経由で評価。
from text_features import (  # noqa: E402
    h8_text_stack,
    h8b_text_stack_full,
    h12_text_stack_full_current_future,
    h12_text_stack_full_current,
    h15_overview_word_stack,
)

FOLD_REGISTRY = {
    "H5": ("業界target encoding(fold内fit)", h5_industry_te),
    "H8": ("テキスト`今後のDX展望`のスタッキング確率(ネストCV)", h8_text_stack),
    "H8b": ("テキスト`企業概要+今後のDX展望`のスタッキング確率(ネストCV)", h8b_text_stack_full),
    "H12": ("テキスト`full/current/future`の3本スタッキング確率(ネストCV)", h12_text_stack_full_current_future),
    "H12fc": ("テキスト`full/current`の2本スタッキング確率(ネストCV)", h12_text_stack_full_current),
    "H15": ("企業概要word TF-IDFスタッキング確率(ネストCV)", h15_overview_word_stack),
}

# 未実装 / 検証のみ
NEEDS_CV = {
    "H6": "アンケート10の符号安定性（B区分・検証のみ）",
}
