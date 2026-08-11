"""LLMラベル特徴（H9）の読み込みとマージ。

`今後のDX展望` を LLM に評価させた3軸スコア。H8(TF-IDFスタッキング)と出所は
同じテキストだが、LLM は意味レベルで「これまでの取り組み」と「今後の方針」を
読み分けられる点が違う（H8のメモ: 「積極」は過去の姿勢の記述に使われるため無力）。

単体の性質（train 742行）:
  llm_dx_execution_score   float 0.4〜4.0  AUC 0.780  購入率 0% → 59%
  llm_external_vendor_need int   0〜4      AUC 0.646  購入率 7.6% → 50%
  llm_cautiousness         int   0〜4      AUC 0.268  購入率 51% → 9.7%（負方向）
3軸は相互に強相関（execution × cautiousness = -0.764）＝実質1〜2次元。

LLM は `購入フラグ` を見ていない行単位の決定的変換なので、fold内fitは不要
（＝ text_features のようなネストCVは要らず、REGISTRY 側に置ける）。
"""
import pandas as pd

LLM_COLS = ["llm_dx_execution_score", "llm_external_vendor_need", "llm_cautiousness"]
ID_COL = "企業ID"
TRAIN_LLM = "data/train_with_llm_3axes.csv"
TEST_LLM = "data/test_with_llm_3axes.csv"

_LLM_CACHE = {}


def load_llm(path=TRAIN_LLM):
    """企業ID を index にした3軸DataFrame を返す（読み込みは1回だけキャッシュ）。"""
    if path not in _LLM_CACHE:
        df = pd.read_csv(path, usecols=[ID_COL] + LLM_COLS)
        assert df[ID_COL].is_unique, f"{path}: 企業IDが重複している"
        assert df[LLM_COLS].notna().all().all(), f"{path}: LLM列に欠損がある"
        _LLM_CACHE[path] = df.set_index(ID_COL)[LLM_COLS]
    return _LLM_CACHE[path]


def add_llm_axes(raw, X, path=TRAIN_LLM):
    """raw[企業ID] を鍵に X へ3軸を結合した新しい DataFrame を返す（X は変更しない）。

    raw: 企業ID列を持つ生データ（X は前処理で企業IDが落ちているため鍵をこちらから取る）
    X  : 特徴量行列。raw と同じ行順・同じ長さであること。
    行順一致ではなく企業IDで引くので、ファイルが差し替わっても壊れない。
    """
    assert len(raw) == len(X), f"raw({len(raw)}) と X({len(X)}) の行数が違う"
    llm = load_llm(path)
    ids = pd.Index(raw[ID_COL].values)
    missing = ids.difference(llm.index)
    assert len(missing) == 0, f"{path} に無い企業IDが {len(missing)} 件: {list(missing[:5])}"

    X = X.copy()
    vals = llm.reindex(ids)
    for c in LLM_COLS:
        X[c] = vals[c].values
    assert X[LLM_COLS].notna().all().all(), "マージ後にNaNが発生した"
    return X
