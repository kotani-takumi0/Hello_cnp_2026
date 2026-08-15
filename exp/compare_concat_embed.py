"""E4 を [組織図] 単独から [組織図 ; 企業概要] の連結に替える案（H26）を多seedで比較する。

これは **E4単体の土俵**での比較で、採否はここでは決まらない。
`exp/holdout_check.py --concat-embed` の合成後ホールドアウトが本判定
（exp020/023/025 は「単体ACCEPT → 合成で負け」の3連敗）。

比較する3本（すべて同一の外側5分割・同一seed・同一 dim/C）:
  org      現行E4 = 組織図 embedding 1024次元 + LR
  concat   [組織図 ; 企業概要] を各1024次元で連結し行を再L2正規化 + LR (2048次元)
  ovw      企業概要 embedding 1024次元 + LR    … 連結の利得が概要側から来ているかの確認

**なぜ「追加」ではなく「差し替え」なのか**:
  企業概要は E0/E3/E6 のどれも使っていない未使用の列。7本目のエキスパートとして
  足すと却下則3（構成員追加は重みが小さくてもコスト）を踏むが、E4 の中身を
  差し替える形なら6本のまま新しい情報源を入れられる。exp026 が3連敗を止めたのも
  この形だった。

**事前登録した停止条件**（結果を見る前に固定する）:
  1. 単体で `decision.verdict` が ACCEPT にならなければ合成に進まない。
  2. concat と org の OOF Spearman が **0.95 以上なら却下**。連結は org を含むので
     相関が高いのは当然だが、0.95 を超えるなら企業概要ブロックが実質何も
     足していないということで、差し替えのコストに見合わない。
     （exp021/exp027 の 0.90 ゲートとは意味が逆なので、線を分けて登録する。
       あちらは「別物であること」の要求、ここは「変化したこと」の要求）
  3. 本判定は holdout_check 5rep の対exp026 ΔF1。**3/5 未満なら REJECT**。
     exp026 自身が対exp019 で 3/5 だったので、それと同じ線に揃える。

  python3 exp/compare_concat_embed.py --n-seeds 10
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
    CONCAT_EMB_C, CONCAT_EMB_DIM, CONCAT_SLUGS, EMB_LR_PARAMS,
    build_embed_model, load_concat_embeddings, load_embeddings,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
CORR_GATE = 0.95


def _scores(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=max(f1s), th=THS[int(np.argmax(f1s))])


def one_seed(y, mats, params, seed):
    """同一の分割で全変種のOOFを作る。分割を共有しないとペア比較にならない。"""
    oof = {k: np.zeros(len(y)) for k in mats}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(next(iter(mats.values())), y):
        for k, m in mats.items():
            fit = build_embed_model(params).fit(m[tr], y[tr])
            oof[k][va] = fit.predict_proba(m[va])[:, 1]
    return oof


def run(n_seeds, dim, C):
    y = pd.read_csv("data/train.csv")["購入フラグ"].values
    params = {**EMB_LR_PARAMS, "C": C}
    mats = {
        "org": load_embeddings("org", dim=dim)[0],
        "concat": load_concat_embeddings(CONCAT_SLUGS, dim=dim)[0],
        "ovw": load_embeddings("overview", dim=dim)[0],
    }
    print(f"行数 {len(y)}  正例率 {y.mean():.4f}   dim={dim}(ブロックごと) C={C}"
          f"  seeds={n_seeds}")
    print(f"  " + "  ".join(f"{k}:{v.shape[1]}次元" for k, v in mats.items()) + "\n")

    rows = {k: [] for k in mats}
    corrs = []
    for seed in range(n_seeds):
        oof = one_seed(y, mats, params, seed)
        for k in rows:
            rows[k].append(_scores(y, oof[k]))
        corrs.append(spearmanr(oof["org"], oof["concat"]).statistic)
        print(f"  seed {seed}: "
              + "  ".join(f"{k} AP {rows[k][-1]['ap']:.4f}" for k in rows)
              + f"  r(org,concat)={corrs[-1]:.3f}", flush=True)

    dfs = {k: pd.DataFrame(v) for k, v in rows.items()}
    print("\n=== 平均±std ===")
    summary = pd.DataFrame({k: d.mean() for k, d in dfs.items()}).T
    summary["ap_std"] = [dfs[k]["ap"].std() for k in summary.index]
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    r = float(np.mean(corrs))
    print(f"\n=== 相関ゲート（事前登録: {CORR_GATE} 以上で却下）===")
    print(f"  org と concat の OOF Spearman: 平均 {r:.4f} "
          f"(最小 {np.min(corrs):.4f} / 最大 {np.max(corrs):.4f})")
    print(f"  → {'違反: 企業概要ブロックが実質効いていない' if r >= CORR_GATE else '통過'}"
          .replace("통過", "通過"))

    print("\n=== ペア判定（基準 = 現行 org 単独）===")
    verdicts = {}
    for k in ("concat", "ovw"):
        v, d = decision.verdict(dfs["org"], dfs[k])
        verdicts[k] = v
        print(decision.format_report(k, f"org -> {k}", dfs["org"], dfs[k], v, d))
        print()

    out = os.path.join(OUT_DIR, "_h26_concat_embed.csv")
    pd.concat([d.assign(model=k, seed=range(len(d))) for k, d in dfs.items()]
              ).to_csv(out, index=False)
    pd.DataFrame([dict(spearman_org_concat=r, corr_gate=CORR_GATE,
                       gate_pass=bool(r < CORR_GATE),
                       verdict_concat=verdicts["concat"],
                       verdict_ovw=verdicts["ovw"])]).to_csv(
        os.path.join(OUT_DIR, "_h26_concat_embed_summary.csv"), index=False)
    print(f"保存: {out}")
    print("\n次: 単体ACCEPT かつ ゲート通過なら "
          "`python3 exp/holdout_check.py --n-reps 5 --concat-embed` で本判定。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--dim", type=int, default=CONCAT_EMB_DIM)
    p.add_argument("--C", type=float, default=CONCAT_EMB_C)
    a = p.parse_args()
    run(a.n_seeds, a.dim, a.C)
