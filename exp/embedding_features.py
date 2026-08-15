"""OpenAI embedding をローカルで扱う層（API は一切叩かない）。

`colab_openai_embed.py` が作った npz を読み、既存のテキスト配管にそのまま挿す。

**設計上の要点 — 文字列ではなく行列を流す**:
  既存の配管（`text_features.nested_text_pred` / `fold_text_preds`,
  `ensemble_experts._nested_text`, `holdout_check.fit_predict_experts`）は
  `txt` を numpy 配列として受け取り `txt[tr]` で fold スライスしているだけなので、
  `txt` を「文字列の1次元配列」から「埋め込みの2次元行列」に差し替えると、
  スライスも fit も predict もそのまま動く。文字列をキーにした辞書引きは
  正規化のズレでバグる（TF-IDF側と embedding側で正規化が違う）ので採らない。

**次元の切り詰め**:
  text-embedding-3 系は Matryoshka 学習済み。先頭 k 次元を取って L2 再正規化すれば
  k 次元モデルとして機能する。742行に対して素の3072次元は明らかに過剰なので、
  `dim` はハイパラとして扱う。API の再課金は要らない。

**ハイパラの決め方**:
  TF-IDF 側（`text_features.TFIDF_PARAMS`）と同じ流儀で、まず単体CVで (dim, C) を
  決め打ちしてから合成に入れる。単体で選んでから合成、の順は守ること
  （exp020/023/025 の3連敗は「単体を強くしたら合成で負けた」なので、
  単体CVの数字を採用根拠にしてはいけない。あくまで設定値を1つに絞るためだけに使う）。

単体CVの下見:
  python3 exp/embedding_features.py --slug dx_outlook
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

DEFAULT_MODEL = "text-embedding-3-large"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(EXP_DIR)
# npz の置き場所。data/ を先に見る（1列あたり11MBあり、素性としてはデータ寄り）。
EMB_DIRS = (os.path.join(_ROOT, "data"), EXP_DIR)

# 列名 -> npz の slug。colab_openai_embed.COLUMNS と対応する。
SLUG_BY_COL = {
    "今後のDX展望": "dx_outlook",
    "企業概要": "overview",
    "組織図": "org",
}

# 埋め込みLRの既定。solver は lbfgs（liblinear の dual は巡回順がランダムで
# 再現性を壊す。text_features.LR_PARAMS のコメント参照）。
# C はまだ暫定値。単体CVで決めてから固定すること。
EMB_LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=5000, random_state=0)
EMB_DIM = 512  # 暫定値。単体CVで決めてから固定すること

# E4(組織図)差し替え用に確定した設定。単体CVグリッド(seed=42)で決め、
# それとは別のseed 0〜9 の10seedペア比較で ACCEPT(明確) を確認済み:
#   ΔAUC +0.0853 (10/10) / ΔAP +0.0434 (10/10) / ΔF1 +0.0547 (10/10)
#   現行TF-IDFとのOOF相関 0.560（exp021 の却下線 0.927 とは別物）
ORG_EMB_DIM, ORG_EMB_C = 1024, 1.0

# E4 を [組織図 ; 企業概要] の連結にする案（H26 / exp029）の設定。
# 単体CVグリッド(seed=42)より: AUC .7056 / AP .4142
#   （org単体 .6796/.3833、overview単体 .6905/.3979 のどちらより上）
# **ORG_EMB_DIM / ORG_EMB_C とわざと同じ値にしてある**。こうすると現行E4との差が
# 「overviewブロックを連結したか」だけになり、dim や C の違いが交絡しない。
# dim=3072 は AP .4143 で +0.0001 ＝ ノイズなので、次元を3倍にする理由が無い。
CONCAT_SLUGS = ("org", "overview")
CONCAT_EMB_DIM, CONCAT_EMB_C = 1024, 1.0

# H30: 3文書間の意味的な不一致を絶対差で表す独立エキスパート。
# quick screen で4候補中もっとも合成OOFの増分が大きかった設定を固定する。
H30_ABSDIFF_SLUGS = ("org", "overview", "dx_outlook")
H30_ABSDIFF_DIM, H30_ABSDIFF_C = 256, 1.0

# 先頭k次元の切り詰め（Matryoshka）が許されるモデル。BERT系はここに入らない。
# 新しい npz は meta に `mrl=0/1` を持つので、そちらが優先される。
MRL_MODELS = ("text-embedding-3-large", "text-embedding-3-small")

_CACHE = {}


def emb_path(slug, model=DEFAULT_MODEL):
    """EMB_DIRS を順に探し、最初に見つかったパスを返す。無ければ先頭候補を返す。"""
    name = f"_emb_{slug}_{model}.npz"
    for d in EMB_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(EMB_DIRS[0], name)


def _load_raw(slug, model):
    key = (slug, model)
    if key not in _CACHE:
        path = emb_path(slug, model)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} が無い。Colab で colab_openai_embed.py を実行して "
                f"npz を取得し、data/ か exp/ に置くこと。"
            )
        d = np.load(path, allow_pickle=True)
        meta = list(d["meta"]) if "meta" in d else []
        _CACHE[key] = (d["train"], d["test"], d["train_ids"], d["test_ids"], meta)
    return _CACHE[key]


def is_matryoshka(model, meta=()):
    """先頭k次元の切り詰めが許されるモデルか。

    npz の meta に `mrl=0/1` があればそれが答え（生成側が知っている）。
    無い古い npz（OpenAI分）はモデル名で判定する。
    """
    for m in meta:
        s = str(m)
        if s.startswith("mrl="):
            return s == "mrl=1"
    return model in MRL_MODELS


def truncate(mat, dim):
    """Matryoshka 切り詰め: 先頭 dim 次元を取って L2 再正規化する。

    **MRL学習済みモデルにしか使ってはいけない。** BERT系の次元には順序の意味が
    無いので、先頭k次元を取るのはただの情報の切り捨てになる。呼び出し元の
    `load_embeddings` が非MRLモデルに対して例外を投げる。
    """
    if dim is None or dim >= mat.shape[1]:
        return mat
    sub = mat[:, :dim]
    return sub / np.linalg.norm(sub, axis=1, keepdims=True)


def load_embeddings(slug, dim=EMB_DIM, model=DEFAULT_MODEL, verify=True):
    """(train行列, test行列) を返す。行順は data/train.csv, data/test.csv と一致。

    verify=True のとき企業IDの並びを CSV と突き合わせる。npz を作り直したときの
    行ズレは静かに効いて全ての比較を壊すので、既定で毎回確認する。
    """
    tr, te, tr_ids, te_ids, meta = _load_raw(slug, model)
    if dim is not None and dim < tr.shape[1] and not is_matryoshka(model, meta):
        raise ValueError(
            f"{model} は Matryoshka ではないので dim={dim} への切り詰めは無意味"
            f"（{tr.shape[1]}次元の先頭だけを取ることになる）。dim=None を渡すこと。"
        )
    if verify:
        for split, ids in (("train", tr_ids), ("test", te_ids)):
            csv = pd.read_csv(f"data/{split}.csv", usecols=["企業ID"])["企業ID"].values
            assert len(csv) == len(ids) and (csv == ids).all(), \
                f"{split} の企業ID並びが npz と一致しない（行ズレ）"
    return truncate(tr, dim), truncate(te, dim)


def load_by_col(col, dim=EMB_DIM, model=DEFAULT_MODEL):
    return load_embeddings(SLUG_BY_COL[col], dim=dim, model=model)


def load_concat_embeddings(slugs=CONCAT_SLUGS, dim=EMB_DIM, model=DEFAULT_MODEL):
    """複数列の埋め込みを横に連結して (train行列, test行列) を返す。

    **各ブロックを先に truncate（= L2正規化）してから連結し、最後に行全体を
    もう一度 L2 正規化する。** 順序に意味がある:

      - ブロックごとの正規化 = どちらの列も同じ重みで入る。片方の文章が長くて
        ノルムが大きい、といった理由で寄与が偏らない。
      - 最後の全体正規化 = 行のノルムが 1 に揃うので、`build_embed_model` の
        「L2正規化済みだから StandardScaler を挟まない」前提と C の意味が
        単一ブロックのときと揃う。これをやらないとノルムが sqrt(len(slugs)) 倍になり、
        同じ C が実質 len(slugs) 倍の緩さになって単体CVの結果と比較できない。

    行順は data/train.csv, data/test.csv と一致する（各 npz を verify 済み）。
    """
    blocks = [load_embeddings(s, dim=dim, model=model) for s in slugs]
    out = []
    for i in range(2):  # 0=train, 1=test
        m = np.hstack([b[i] for b in blocks])
        out.append(m / np.linalg.norm(m, axis=1, keepdims=True))
    return out[0], out[1]


def load_absdiff_embeddings(dim=H30_ABSDIFF_DIM, model=DEFAULT_MODEL):
    """H30の3文書間絶対差を連結し、行L2正規化して返す。"""
    blocks = {
        slug: load_embeddings(slug, dim=dim, model=model)
        for slug in H30_ABSDIFF_SLUGS
    }
    out = []
    for split in range(2):
        org = blocks["org"][split]
        overview = blocks["overview"][split]
        dx = blocks["dx_outlook"][split]
        matrix = np.hstack((np.abs(dx - org), np.abs(dx - overview),
                            np.abs(org - overview)))
        out.append(matrix / np.linalg.norm(matrix, axis=1, keepdims=True))
    return out[0], out[1]


def build_embed_model(params=None):
    """文字列ではなく埋め込み行列を受け取る LR。`text_features.build_model` の代替。

    埋め込みは L2 正規化済みなので StandardScaler は挟まない
    （次元ごとに標準化するとノルム構造が壊れ、正則化の効き方が読めなくなる）。
    """
    return make_pipeline(LogisticRegression(**(params or EMB_LR_PARAMS)))


def cross_similarity(slug_a, slug_b, dim=None, model=DEFAULT_MODEL):
    """2つのテキスト列の意味的な近さ（コサイン）を train/test 分の1次元で返す。

    例: cross_similarity("overview", "dx_outlook") = 「本業と今後の展望の乖離」。
    char n-gram TF-IDF が構造的に作れない量で、ラベルを見ない行単位の決定的変換
    なので fold 内 fit も不要（=リークしない）。
    """
    a_tr, a_te = load_embeddings(slug_a, dim=dim, model=model)
    b_tr, b_te = load_embeddings(slug_b, dim=dim, model=model, verify=False)
    return (a_tr * b_tr).sum(axis=1), (a_te * b_te).sum(axis=1)


def _cv_scores(mat, y, params, seed=42, n_splits=5):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(mat, y):
        m = build_embed_model(params).fit(mat[tr], y[tr])
        oof[va] = m.predict_proba(mat[va])[:, 1]
    ths = np.arange(0.05, 0.95, 0.005)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, oof), ap=average_precision_score(y, oof),
                f1=f1s[b], th=ths[b]), oof


def main():
    """単体CVで (dim, C) の当たりを付ける。合成の採否判断には使わない。"""
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default="dx_outlook", choices=list(SLUG_BY_COL.values()))
    p.add_argument("--concat", nargs="+", default=None,
                   choices=list(SLUG_BY_COL.values()),
                   help="複数列を連結した1本として評価する（例: --concat org overview）")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dims", nargs="+", type=int, default=[256, 512, 1024, 3072])
    p.add_argument("--cs", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    y = pd.read_csv("data/train.csv")["購入フラグ"].values
    name = "+".join(a.concat) if a.concat else a.slug
    if a.concat:
        # 連結は dim がブロックごとに効くので、行列は dim ごとに作り直す
        full, _ = load_embeddings(a.concat[0], dim=None, model=a.model)
        print(f"{name}: ブロック{len(a.concat)}本 x 最大{full.shape[1]}次元"
              f"  正例率 {y.mean():.4f}")
    else:
        full, _ = load_embeddings(a.slug, dim=None, model=a.model)
        print(f"{name}: {full.shape}  正例率 {y.mean():.4f}")

    # 非MRLモデル（BERT系）は次元をハイパラにできない。素の次元1本だけ回す。
    meta = _load_raw(a.concat[0] if a.concat else a.slug, a.model)[4]
    dims = a.dims if is_matryoshka(a.model, meta) else [None]
    if dims == [None]:
        print(f"  ({a.model} は非Matryoshka: 切り詰めずに{full.shape[1]}次元で評価)")

    rows = []
    for dim in dims:
        if dim is not None and dim > full.shape[1]:
            continue
        mat = (load_concat_embeddings(tuple(a.concat), dim=dim, model=a.model)[0]
               if a.concat else truncate(full, dim))
        for c in a.cs:
            s, _ = _cv_scores(mat, y, {**EMB_LR_PARAMS, "C": c}, seed=a.seed)
            rows.append(dict(dim=dim, C=c, **s))
            print(f"  dim={dim:<5} C={c:<5} AUC {s['auc']:.4f}  AP {s['ap']:.4f}"
                  f"  F1 {s['f1']:.4f} @ {s['th']:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values("ap", ascending=False)
    out = os.path.join(EXP_DIR, f"_emb_cv_{name.replace('+', '_')}.csv")
    df.to_csv(out, index=False)
    print(f"\n=== AP上位 ===\n{df.head(5).to_string(index=False)}")
    print(f"保存: {out}")
    print("\n参考（既存TF-IDF+LR単体, seed=42）: "
          "DX展望 AUC .7894 / AP .5063 / F1 .5916, 組織図 AUC .6095 / AP .3437, "
          "企業概要 AUC .650")


if __name__ == "__main__":
    main()
