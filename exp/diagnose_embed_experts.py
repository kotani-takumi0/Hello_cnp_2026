"""埋め込みエキスパート候補が既存6本とどれだけ直交しているかを見る。

`ensemble_experts.compute_expert_preds` のキャッシュ（`_experts_seed{seed}.npz`）に
入っている既存6本のOOFと、埋め込みから作った候補のOOFを **同一の外側5分割** で
突き合わせる。分割は `StratifiedKFold(5, shuffle=True, random_state=seed)` で
y だけから決まるので、既存キャッシュと同じ並びになる（再学習は候補側だけで済む）。

見るのは2つ:
  1. 単体AP が正例率 0.241 をどれだけ上回るか（`insights`の「直交≠信号あり」則）
  2. 既存6本との相関。exp021 は E3 と 0.927 ＝ 劣化コピーで REJECT された

候補:
  E4b_org_emb       組織図 embedding+LR   … 既存 E4 の **差し替え** 候補
  E5_overview_emb   企業概要 embedding+LR … **新規追加** 候補（未使用の情報源）
  E3b_dx_emb        DX展望 embedding+LR   … 既存 E3 の差し替え候補
  E5x_cross_sim     cos(企業概要, DX展望) … 学習なしの1次元（本業と展望の乖離）

  python3 exp/diagnose_embed_experts.py --seed 42
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
from embedding_features import (  # noqa: E402
    EMB_LR_PARAMS, build_embed_model, cross_similarity, load_embeddings,
)
from ensemble_experts import EXPERTS, OUT_DIR  # noqa: E402

THS = np.arange(0.05, 0.95, 0.005)

# 単体CVグリッド（embedding_features.py, seed=42）から列ごとに1組ずつ固定。
CANDIDATES = {
    "E4b_org_emb": ("org", 1024, 1.0),
    "E5_overview_emb": ("overview", 1024, 0.3),
    "E3b_dx_emb": ("dx_outlook", 512, 1.0),
}


def _scores(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1s[b], th=THS[b])


def _oof(emb, y, C, seed):
    out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = build_embed_model({**EMB_LR_PARAMS, "C": C}).fit(emb[tr], y[tr])
        out[va] = m.predict_proba(emb[va])[:, 1]
    return out


def run(seed):
    cache = os.path.join(OUT_DIR, f"_experts_seed{seed}.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} が無い。先に ensemble_experts.py --seed {seed} を実行する")
    d = np.load(cache)
    y = d["y"]
    oof = {n: d[f"oof_{n}"] for n in EXPERTS}

    for name, (slug, dim, C) in CANDIDATES.items():
        emb, _ = load_embeddings(slug, dim=dim)
        oof[name] = _oof(emb, y, C, seed)
        print(f"  {name} 済み (dim={dim}, C={C})", flush=True)

    # 学習を伴わない1次元。符号は「似ているほど購入」を仮定して素のcosを入れる。
    sim, _ = cross_similarity("overview", "dx_outlook")
    oof["E5x_cross_sim"] = sim

    names = list(EXPERTS) + list(CANDIDATES) + ["E5x_cross_sim"]
    table = pd.DataFrame({n: _scores(y, oof[n]) for n in names}).T
    print(f"\n=== 単体OOFスコア (seed={seed})   正例率 {y.mean():.4f} ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("  ※ AP が正例率 0.2412 をどれだけ上回るかを見る"
          "（E2 は 0.4283 = 1.8倍で当たり、企業名は 0.27 でハズレだった）")

    P = np.column_stack([oof[n] for n in names])
    corr = pd.DataFrame(spearmanr(P).correlation, index=names, columns=names)
    print("\n=== OOF予測のSpearman相関 ===")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== 候補ごとの『既存6本との最大相関』（低いほど直交）===")
    for n in list(CANDIDATES) + ["E5x_cross_sim"]:
        s = corr.loc[n, list(EXPERTS)].abs().sort_values(ascending=False)
        print(f"  {n:16s} 最大 |r| = {s.iloc[0]:.3f} ({s.index[0]})"
              f"  次点 {s.iloc[1]:.3f} ({s.index[1]})")

    out = os.path.join(OUT_DIR, f"_emb_diag_corr_seed{seed}.csv")
    corr.to_csv(out)
    table.to_csv(os.path.join(OUT_DIR, f"_emb_diag_scores_seed{seed}.csv"))
    print(f"\n保存: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args().seed)
