"""テキスト前処理の比較ラボ（テキスト単体モデルで前処理だけを評価する）。

なぜテキスト単体で測るか:
  GBDT に混ぜてから比較すると、前処理の差がテーブル特徴に薄められて見えなくなる。
  「テキストから信号をどれだけ引き出せたか」を測りたいので、
  TF-IDF + LogisticRegression 単体の OOF AUC/AP で前処理だけを比較する。
  ここで伸びた分がスタッキング経由で GBDT に渡る上限になる。

使い方: python3 exp/text_lab.py
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

TARGET = "購入フラグ"
TEXT_COL = "今後のDX展望"
OVERVIEW_COL = "企業概要"
SEEDS = (0, 1, 2)
N_SPLITS = 5
N_JOBS = 8

# ---------------------------------------------------------------- 正規化

# 数字は「4,539名」「792名」など企業固有の値。テキストからは従業員数として
# 既にテーブル側に入っており、char n-gram では固有値の丸暗記になるだけ。
_NUM = re.compile(r"[0-9]+(?:[,.][0-9]+)*")
_SPACE = re.compile(r"[\s　]+")


def normalize(s: str) -> str:
    """NFKC → 数字マスク → 空白圧縮。日本語テキストの最低限の下ごしらえ。"""
    s = unicodedata.normalize("NFKC", s)  # 全角英数→半角, ｶﾅ→カナ を統一
    s = _NUM.sub("0", s)                  # 数値の桁違いを1トークンに潰す
    s = _SPACE.sub(" ", s)                # 改行・全角スペースの揺れを吸収
    return s.strip()


def normalize_keep_layout(s: str) -> str:
    """段落構造(\n\n)を保ったまま正規化する版。"""
    parts = re.split(r"\n\s*\n", s)
    return "\n\n".join(normalize(p) for p in parts if p.strip())


# ---------------------------------------------------------------- 構造分割

# この文書は全て「これまでは○○だった / 今後は△△する」の対比構造で書かれている。
# 96% の文書が両方のマーカーを持つ。同じ「慎重」でも過去文なら現状説明、
# 未来文なら購入意欲の低さを意味する = 意味が逆。char n-gram はこれを区別できない。
PAST_MARK = ("これまで", "従来", "現状", "既存", "とどま", "に過ぎ", "でした",
             "ました", "きました", "残っており", "が実情")
FUTURE_MARK = ("今後", "まいります", "計画", "方針", "予定", "検討",
               "していく", "図る", "目指", "する考え", "拡大し", "整備し")

_SENT = re.compile(r"(?<=。)")


def split_sentences(text: str):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def tense_of(sent: str) -> str:
    """文を過去(現状説明)/未来(計画)/その他 に分類する。"""
    p = any(m in sent for m in PAST_MARK)
    f = any(m in sent for m in FUTURE_MARK)
    if f and not p:
        return "future"
    if p and not f:
        return "past"
    return "both"


def tense_parts(text: str):
    """(過去文の連結, 未来文の連結) を返す。both は両方に入れる。"""
    past, future = [], []
    for s in split_sentences(text):
        t = tense_of(s)
        if t in ("past", "both"):
            past.append(s)
        if t in ("future", "both"):
            future.append(s)
    return " ".join(past), " ".join(future)


# 目的変数は「DX教育商材の購入」。教育・人材育成に触れた文だけを抜き出せば
# ノイズ(セキュリティ, 設備投資などの無関係な段落)を落とせるはず、という仮説。
EDU_MARK = ("教育", "研修", "人材", "スキル", "リテラシー", "リスキリング",
            "ラーニング", "ワークショップ", "OJT", "育成", "講師", "カリキュラム",
            "アカデミー", "セミナー", "学習")


def edu_part(text: str) -> str:
    return " ".join(s for s in split_sentences(text)
                    if any(m in s for m in EDU_MARK))


# ---------------------------------------------------------------- 評価

def make_model(**kw):
    params = dict(analyzer="char_wb", ngram_range=(1, 3), min_df=5,
                  sublinear_tf=True)
    params.update(kw)
    return TfidfVectorizer(**params)


# liblinear の双対解法は n_samples(742) << n_features(数万) の疎行列で lbfgs の
# 20倍速い。解は実質同じ（L2ロジスティック回帰）なので比較目的には十分。
# random_state 必須: dual 座標降下法は座標の巡回順をランダムにシャッフルするため、
# 未指定だと同一入力でも確率が 1e-6 オーダーで揺れる。詳細は
# text_features.LR_PARAMS のコメントを参照。
LR_PARAMS = dict(C=1.0, solver="liblinear", dual=True, max_iter=3000,
                 random_state=0)


def _one_fold(build, y, tr, va):
    Xtr, Xva = build(tr, va)
    m = LogisticRegression(**LR_PARAMS).fit(Xtr, y[tr])
    return va, m.predict_proba(Xva)[:, 1]


def evaluate(build, y, label):
    """build(idx_tr, idx_va) -> (Xtr, Xva) を受け取り OOF 指標を返す。
    ベクトライザの fit は必ず fold の train 側だけで行う（語彙リーク防止）。"""
    jobs = [(seed, tr, va) for seed in SEEDS
            for tr, va in StratifiedKFold(N_SPLITS, shuffle=True,
                                          random_state=seed).split(np.zeros(len(y)), y)]
    out = Parallel(n_jobs=N_JOBS)(delayed(_one_fold)(build, y, tr, va)
                                  for _, tr, va in jobs)
    aucs, aps = [], []
    for k in range(len(SEEDS)):
        oof = np.zeros(len(y))
        for va, p in out[k * N_SPLITS:(k + 1) * N_SPLITS]:
            oof[va] = p
        aucs.append(roc_auc_score(y, oof))
        aps.append(average_precision_score(y, oof))
    print(f"  {label:38s} AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}   "
          f"AP {np.mean(aps):.4f}±{np.std(aps):.4f}", flush=True)
    return np.mean(aucs), np.mean(aps)


def single_field(texts, **kw):
    """1つのテキスト列を1つの TF-IDF に通す builder。"""
    arr = np.asarray(texts, dtype=object)

    def build(tr, va):
        v = make_model(**kw)
        return v.fit_transform(arr[tr]), v.transform(arr[va])
    return build


def multi_field(fields, **kw):
    """複数フィールドを別々にベクトル化して横結合する builder。
    各フィールドが独立した語彙空間を持つので「過去の慎重」と「未来の慎重」に
    別々の重みが付く（同じ vectorizer に混ぜると区別できない）。"""
    arrs = [np.asarray(f, dtype=object) for f in fields]

    def build(tr, va):
        Xt, Xv = [], []
        for a in arrs:
            v = make_model(**kw)
            Xt.append(v.fit_transform(a[tr]))
            Xv.append(v.transform(a[va]))
        return hstack(Xt).tocsr(), hstack(Xv).tocsr()
    return build


def main():
    df = pd.read_csv("data/train.csv")
    y = df[TARGET].values
    raw = df[TEXT_COL].fillna("").tolist()
    ovw = df[OVERVIEW_COL].fillna("").tolist()

    norm = [normalize(t) for t in raw]
    ovw_n = [normalize(t) for t in ovw]
    past = [tense_parts(t)[0] for t in norm]
    future = [tense_parts(t)[1] for t in norm]
    edu = [edu_part(t) for t in norm]

    print(f"平均文字数: 原文 {np.mean([len(t) for t in raw]):.0f} / "
          f"過去 {np.mean([len(t) for t in past]):.0f} / "
          f"未来 {np.mean([len(t) for t in future]):.0f} / "
          f"教育 {np.mean([len(t) for t in edu]):.0f}")
    print(f"教育文が空の行: {sum(1 for t in edu if not t)} / {len(edu)}\n")

    print("【0】現状の設定（text_features.py と同じ）")
    base = evaluate(single_field(raw), y, "raw, char_wb(1,3), min_df=5")

    print("\n【1】正規化を入れる（NFKC + 数字マスク + 空白圧縮）")
    evaluate(single_field(norm), y, "normalized")

    print("\n【2】n-gram 幅を変える（日本語は3文字では短すぎる）")
    for ng in [(2, 3), (2, 4), (2, 5), (3, 5)]:
        evaluate(single_field(norm, ngram_range=ng), y, f"normalized, char_wb{ng}")

    print("\n【3】analyzer: char_wb と char の違い")
    evaluate(single_field(norm, analyzer="char", ngram_range=(2, 4)), y,
             "normalized, char(2,4)")

    print("\n【4】語彙の枝刈り（全文書に出る定型文を落とす）")
    for kw in [dict(min_df=2), dict(min_df=10), dict(max_df=0.9), dict(max_df=0.7)]:
        evaluate(single_field(norm, ngram_range=(2, 4), **kw), y,
                 f"normalized, char_wb(2,4), {kw}")

    print("\n【5】構造分割: 時制で分けて別々にベクトル化")
    evaluate(single_field(future, ngram_range=(2, 4)), y, "未来文のみ")
    evaluate(single_field(past, ngram_range=(2, 4)), y, "過去文のみ")
    evaluate(multi_field([past, future], ngram_range=(2, 4)), y,
             "過去 | 未来 を別空間で結合")

    print("\n【6】構造分割: 教育・人材の文だけ（目的変数に直結する話題）")
    evaluate(single_field(edu, ngram_range=(2, 4)), y, "教育文のみ")
    evaluate(multi_field([edu, future], ngram_range=(2, 4)), y, "教育 | 未来")
    evaluate(multi_field([past, future, edu], ngram_range=(2, 4)), y,
             "過去 | 未来 | 教育")

    print("\n【7】企業概要の足し方: 連結 vs 別空間")
    cat = [a + " " + b for a, b in zip(ovw_n, norm)]
    evaluate(single_field(cat, ngram_range=(2, 4)), y, "企業概要+展望 を1本に連結(現H8b)")
    evaluate(multi_field([ovw_n, norm], ngram_range=(2, 4)), y, "企業概要 | 展望 を別空間")
    evaluate(multi_field([ovw_n, past, future, edu], ngram_range=(2, 4)), y,
             "概要 | 過去 | 未来 | 教育")

    print(f"\n（基準）現状 AUC {base[0]:.4f} / AP {base[1]:.4f}")


if __name__ == "__main__":
    main()
