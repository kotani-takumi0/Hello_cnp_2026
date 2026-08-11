"""テキスト列からの特徴生成（H8: 自由記述スタッキング）。

`今後のDX展望` は日本語自由記述（平均852文字）。単体の char n-gram TF-IDF +
ロジスティック回帰で OOF AUC≈0.79 / AP≈0.51 出る強い信号で、しかも
アンケート1〜11 とほぼ無相関（|r|<0.08）＝ テーブル特徴と独立。

これを1本の確率に落として GBDT に渡す（スタッキング）。
リーク防止のため **ネストCV** で作る:
  - 外側 fold の train 部分 → 内側 CV の OOF 予測を割り当て
  - 外側 fold の valid 部分 → train 部分全体で学習したモデルで予測
`harness.run_cv` の fold_transform 経由で呼ばれるため、外側の分割は
harness 側が担保する。
"""
import numpy as np
import pandas as pd
import re
import unicodedata
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

TEXT_COL = "今後のDX展望"
OVERVIEW_COL = "企業概要"
INNER_SPLITS = 5
INNER_SEED = 0  # 内側CVは固定。外側分割が変われば入力が変わるので多様性は保たれる

# 日本語は分かち書きなしで char_wb。C(0.3〜30)・min_df(2〜10) はほぼ無感応だったので
# 素の値を採用（単体CV: DX展望のみ AUC.786/AP.509、+企業概要 AUC.806/AP.560）。
TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(1, 3), min_df=5, sublinear_tf=True)
OVERVIEW_TFIDF_PARAMS = dict(
    analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2),
    min_df=3, max_df=0.9, sublinear_tf=True
)
# n_samples(742) << n_features の疎行列なので liblinear dual が速い。
# L2 logistic regression で、H10ラボの baseline AP は既存H8ログと一致している。
# random_state 必須: liblinear の dual 座標降下法は座標の巡回順をランダムに
# シャッフルするため、未指定だと同一入力でも確率が 1e-6 オーダーで揺れる。
# このスコアは下流LightGBMで gain 単独首位（2位の2.3倍）なので、その揺れが
# 分割点→OOF閾値→境界行のラベル と増幅され、再実行するだけで提出が2行変わる。
LR_PARAMS = dict(C=1.0, solver="liblinear", dual=True, max_iter=3000,
                 random_state=0)

_TEXT_CACHE = {}
_TENSE_CACHE = {}
_OVERVIEW_TOKEN_CACHE = {}

_NUM = re.compile(r"[0-9]+(?:[,.][0-9]+)*")
_SPACE = re.compile(r"[\s　]+")
_SENT = re.compile(r"(?<=。)")
_TOKEN_KEEP = re.compile(r"[0-9A-Za-zぁ-んァ-ヶ一-龥]+")
_HIRAGANA_1 = re.compile(r"^[ぁ-ん]$")
_JANOME = None

CURRENT_MARK = (
    "これまで", "従来", "現状", "既存", "とどま", "に過ぎ", "でした",
    "ました", "きました", "残っており", "が実情", "限定的", "不十分",
)
FUTURE_MARK = (
    "今後", "まいります", "計画", "方針", "予定", "検討", "していく",
    "図る", "目指", "する考え", "拡大し", "整備し", "強化", "推進",
)


def load_text(cols=(TEXT_COL,), path="data/train.csv"):
    """指定テキスト列を空白結合して index 付きで返す（harness の X.index と一致）。"""
    key = (path, tuple(cols))
    if key not in _TEXT_CACHE:
        df = pd.read_csv(path)
        s = df[cols[0]].fillna("")
        for c in cols[1:]:
            s = s + " " + df[c].fillna("")
        _TEXT_CACHE[key] = s
    return _TEXT_CACHE[key]


def _normalize(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = _NUM.sub("0", s)
    s = _SPACE.sub(" ", s)
    return s.strip()


def _tokenizer():
    global _JANOME
    if _JANOME is None:
        _JANOME = Tokenizer()
    return _JANOME


def overview_tokenize(text):
    """企業概要向けの粗い日本語word tokenizer。

    EDAでは企業概要は具体的な業界語・事業語が効いたため、charではなくword TF-IDF
    を優先する。1文字ひらがななどの機能語は落とし、名詞/動詞/形容詞などの表層語を
    残す。
    """
    out = []
    for tok in _tokenizer().tokenize(_normalize(text)):
        surface = tok.surface
        if not _TOKEN_KEEP.search(surface) or _HIRAGANA_1.match(surface):
            continue
        base = tok.base_form if tok.base_form != "*" else surface
        out.append(base)
    return out


def tokenize_overview_texts(texts):
    return np.array([" ".join(overview_tokenize(t)) for t in texts], dtype=object)


def load_overview_tokenized(path="data/train.csv"):
    """企業概要を一度だけJanome分かち書きしてキャッシュする。"""
    if path not in _OVERVIEW_TOKEN_CACHE:
        df = pd.read_csv(path)
        _OVERVIEW_TOKEN_CACHE[path] = tokenize_overview_texts(
            df[OVERVIEW_COL].fillna("").astype(str).values
        )
    return _OVERVIEW_TOKEN_CACHE[path]


def _split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _tense_of(sent):
    current = any(m in sent for m in CURRENT_MARK)
    future = any(m in sent for m in FUTURE_MARK)
    if future and not current:
        return "future"
    if current and not future:
        return "current"
    return "both"


def _tense_parts(text):
    current, future = [], []
    for sent in _split_sentences(_normalize(text)):
        tense = _tense_of(sent)
        if tense in ("current", "both"):
            current.append(sent)
        if tense in ("future", "both"):
            future.append(sent)
    return " ".join(current), " ".join(future)


def load_tense_texts(path="data/train.csv"):
    """full/current/future の3系列を返す。full は現H8と同じ原文を使う。"""
    if path not in _TENSE_CACHE:
        df = pd.read_csv(path)
        full = df[TEXT_COL].fillna("").astype(str)
        parts = full.map(_tense_parts)
        current = parts.map(lambda x: x[0])
        future = parts.map(lambda x: x[1])
        _TENSE_CACHE[path] = {
            "full": full,
            "current": current,
            "future": future,
        }
    return _TENSE_CACHE[path]


def build_model():
    return make_pipeline(TfidfVectorizer(**TFIDF_PARAMS), LogisticRegression(**LR_PARAMS))


def build_overview_model():
    return make_pipeline(
        TfidfVectorizer(**OVERVIEW_TFIDF_PARAMS),
        LogisticRegression(**LR_PARAMS),
    )


def nested_text_pred(txt_tr, y_tr, txt_va):
    """(train部分のOOF予測, valid部分の予測) を返す。train統計のみ使用＝リーク無し。"""
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1]


def fold_text_preds(txt_tr, y_tr, txt_va, txt_te, inner_seed=INNER_SEED):
    """`nested_text_pred` に test 予測を足した版。

    (train部分のOOF, valid予測, test予測) を返す。test も「fold train 全体で
    学習したモデル」で予測する＝ valid と同じ作り方なので、train側(内側OOF)と
    test側で特徴の分布が揃う。外側 fold ごとに呼び直すこと。

    アンサンブルではこれを **fold につき1回** 呼び、結果を全ベースモデルで共有する。
    テキストモデルは下流のGBDTに依存しないので、モデル数を増やしても
    ここの計算量は増えない。
    """
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=inner_seed)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1], full.predict_proba(txt_te)[:, 1]


def fold_overview_preds(txt_tr, y_tr, txt_va, txt_te, inner_seed=INNER_SEED):
    """企業概要 word TF-IDF+LR のfold内スタッキング予測を返す。

    入力は `tokenize_overview_texts` 済みの空白区切り文字列を想定する。
    """
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=inner_seed)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_overview_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_overview_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1], full.predict_proba(txt_te)[:, 1]


def _stack(cols, name):
    """指定テキスト列群からスタッキング確率1本を作る fold_transform を返す。"""
    def _f(Xtr, ytr, Xva):
        text = load_text(cols)
        Xtr, Xva = Xtr.copy(), Xva.copy()
        oof, va = nested_text_pred(text.loc[Xtr.index].values, ytr,
                                   text.loc[Xva.index].values)
        Xtr[name] = oof
        Xva[name] = va
        return Xtr, Xva
    return _f


def _stack_text_series(series_by_name):
    """複数テキスト系列をそれぞれ別LRスコアにして追加する fold_transform。"""
    def _f(Xtr, ytr, Xva):
        Xtr, Xva = Xtr.copy(), Xva.copy()
        for name, series in series_by_name.items():
            oof, va = nested_text_pred(series.loc[Xtr.index].values, ytr,
                                       series.loc[Xva.index].values)
            Xtr[name] = oof
            Xva[name] = va
        return Xtr, Xva
    return _f


def _tense_stack(names):
    tense = load_tense_texts()
    return _stack_text_series({f"DX展望_{n}_購入確率": tense[n] for n in names})


# H8 [テキスト自由記述]: `今後のDX展望` の購入確率をスタッキングで1本の特徴に。
# 「慎重」を含むだけで購入率 44%→12.6% に落ちるなど極性語が強く効く一方、「積極」は
# 無力（過去の姿勢の記述に使われるため）。単語有無フラグでは拾えない文脈依存を、
# 教師ありモデルに任せて1次元に圧縮する。
h8_text_stack = _stack((TEXT_COL,), "DX展望_購入確率")

# H8b: `企業概要` を結合した版。企業概要"単独"は AUC.650 と弱いが、DX展望と結合すると
# AUC .786→.806 / AP .509→.560 と伸びる（事業内容が展望の解釈文脈になる）。
# ただし業界情報を含むため既存カテゴリ列と重複する可能性があり、GBDTへの増分は要検証。
h8b_text_stack_full = _stack((OVERVIEW_COL, TEXT_COL), "テキスト_購入確率")

# H12: 文章構造を使う本命版。full は捨てず、full/current/future を別々の
# TF-IDF+LRスタッキング確率に落として、テーブルモデルへ追加する。
h12_text_stack_full_current_future = _tense_stack(("full", "current", "future"))
h12_text_stack_full_current = _tense_stack(("full", "current"))


def h15_overview_word_stack(Xtr, ytr, Xva):
    """H15: `企業概要` word TF-IDF+LR の購入確率を1本追加する。"""
    text = pd.Series(load_overview_tokenized(), index=load_text((OVERVIEW_COL,)).index)
    Xtr, Xva = Xtr.copy(), Xva.copy()
    oof, va = nested_overview_pred(text.loc[Xtr.index].values, ytr,
                                   text.loc[Xva.index].values)
    Xtr["企業概要_購入確率"] = oof
    Xva["企業概要_購入確率"] = va
    return Xtr, Xva


def nested_overview_pred(txt_tr, y_tr, txt_va):
    """H15のfold_transform用。train部分OOFとvalid予測を返す。"""
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_overview_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_overview_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1]
