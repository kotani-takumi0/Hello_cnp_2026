"""テキスト単体の土俵で TF-IDF+LR と OpenAI embedding+LR を多seedペア比較する。

これは **E3（および E0 のスコア列）の中身をどれにするか** を決めるための実験で、
合成の採否実験ではない。判定は `decision.verdict`（AP主・AUC確認・F1ガードレール）。

比較する3本（すべて同一の外側5分割・同一seed）:
  tfidf   現行 `text_features.build_model()`（char_wb 1-3gram + LR）
  embed   `embedding_features.build_embed_model()`（3072次元を dim に切り詰め + LR）
  blend   上2本の **logit平均**。埋め込みが現行を置き換えられなくても、
          直交していれば混ぜたスコアが両者に勝つ可能性がある。E0 に渡るのは
          1本のスコアなので、混ぜるならここで混ぜるのが自然な場所になる。

埋め込み側のハイパラは **事前に固定**して渡す（既定 dim=512 / C=1.0）。
単体CVグリッド(`embedding_features.py`)の argmax をそのまま採ると、20セルから
最大値を拾った分だけ楽観が乗るため。グリッドは AUC .786〜.796 とほぼ平坦だったので
この固定は低リスク。

  python3 exp/compare_embed_text.py --slug dx_outlook --n-seeds 10
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision  # noqa: E402
from embedding_features import (  # noqa: E402
    EMB_LR_PARAMS, SLUG_BY_COL, build_embed_model, load_embeddings,
)
from meta_blend import to_logit  # noqa: E402
from organization_features import ORG_COL, build_org_model  # noqa: E402
from text_features import (  # noqa: E402
    OVERVIEW_COL, TEXT_COL, build_model, build_overview_model,
    load_overview_tokenized,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
COL_BY_SLUG = {v: k for k, v in SLUG_BY_COL.items()}


def _tfidf_inputs(slug, train):
    """現行のテキストモデルと、その入力（列ごとに前処理が違う）を返す。

    企業概要だけ Janome 分かち書き済みの word TF-IDF、組織図は char(2,6)、
    DX展望は char_wb(1,3)。ここを取り違えると現行の再現にならない。
    """
    if slug == "dx_outlook":
        return train[TEXT_COL].fillna("").astype(str).values, build_model
    if slug == "org":
        return train[ORG_COL].fillna("").astype(str).values, build_org_model
    if slug == "overview":
        return load_overview_tokenized("data/train.csv"), build_overview_model
    raise ValueError(slug)


def _scores(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1s[b], th=THS[b])


def one_seed(y, txt, model_fn, emb, emb_params, seed):
    """同一分割で tfidf / embed の OOF を作り、logit平均も返す。"""
    oof = {k: np.zeros(len(y)) for k in ("tfidf", "embed")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = model_fn().fit(txt[tr], y[tr])
        oof["tfidf"][va] = m.predict_proba(txt[va])[:, 1]
        e = build_embed_model(emb_params).fit(emb[tr], y[tr])
        oof["embed"][va] = e.predict_proba(emb[va])[:, 1]
    z = to_logit(np.column_stack([oof["tfidf"], oof["embed"]]))
    oof["blend"] = 1.0 / (1.0 + np.exp(-z.mean(axis=1)))
    return oof


def run(slug, n_seeds, dim, C, model):
    train = pd.read_csv("data/train.csv")
    y = train["購入フラグ"].values
    txt, model_fn = _tfidf_inputs(slug, train)
    emb, _ = load_embeddings(slug, dim=dim, model=model)
    emb_params = {**EMB_LR_PARAMS, "C": C}

    print(f"列: {COL_BY_SLUG[slug]}  行数 {len(y)}  正例率 {y.mean():.4f}")
    print(f"embed: dim={dim} C={C} model={model}   seeds={n_seeds}\n")

    rows = {k: [] for k in ("tfidf", "embed", "blend")}
    corrs = []
    for seed in range(n_seeds):
        oof = one_seed(y, txt, model_fn, emb, emb_params, seed)
        for k in rows:
            rows[k].append(_scores(y, oof[k]))
        corrs.append(spearmanr(oof["tfidf"], oof["embed"]).statistic)
        print(f"  seed {seed}: "
              + "  ".join(f"{k} AP {rows[k][-1]['ap']:.4f}" for k in rows)
              + f"  r={corrs[-1]:.3f}", flush=True)

    dfs = {k: pd.DataFrame(v) for k, v in rows.items()}
    print("\n=== 平均±std ===")
    summary = pd.DataFrame({k: d.mean() for k, d in dfs.items()}).T
    summary["ap_std"] = [dfs[k]["ap"].std() for k in summary.index]
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== tfidf と embed の OOF Spearman相関 ===")
    print(f"  平均 {np.mean(corrs):.4f}  (最小 {np.min(corrs):.4f} / "
          f"最大 {np.max(corrs):.4f})")
    print("  参考: exp021 で REJECT した H9(LLM3軸) は E3 と 0.927 だった。"
          "\n        高いほど『同じテキストの別表現』＝置き換えても合成は動かない。")

    print("\n=== ペア判定（基準 = 現行 tfidf）===")
    for k in ("embed", "blend"):
        v, d = decision.verdict(dfs["tfidf"], dfs[k])
        print(decision.format_report(k, f"tfidf -> {k}", dfs["tfidf"], dfs[k], v, d))
        print()

    out = os.path.join(OUT_DIR, f"_emb_cmp_{slug}.csv")
    pd.concat([d.assign(model=k, seed=range(len(d))) for k, d in dfs.items()]
              ).to_csv(out, index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default="dx_outlook", choices=list(COL_BY_SLUG))
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--model", default="text-embedding-3-large")
    a = p.parse_args()
    run(a.slug, a.n_seeds, a.dim, a.C, a.model)
