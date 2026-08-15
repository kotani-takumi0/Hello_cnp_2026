"""H41: E4 の埋め込みを OpenAI から日本語BERT系に差し替える案を多seedで比較する。

これは **E4単体の土俵**での比較で、採否はここでは決まらない
（exp020/023/025 の3連敗、H31 の GAM 置換失敗はいずれも「単体で勝って合成で負けた」）。
本判定は `exp/holdout_check.py --concat-embed --emb-model <key>` の合成後ホールドアウト。

比較する系列（すべて同一の外側5分割・同一seed・同一C）:
  openai   現行E4 = [組織図;企業概要] の OpenAI text-embedding-3-large 各1024次元
  bert     同じ2列を日本語BERT系で埋め込み、同じ手順で連結（次元は素のまま）
  bert_org / bert_ovw   連結の利得がどちらの列から来ているかの確認用（--per-col）

**なぜ「追加」ではなく「差し替え」で測るのか**:
  却下則3「構成員追加は重みが小さくてもコスト」。同じ2列を見る2本目のエキスパートを
  9本目として足す形は、情報源が重複しているぶん最初から不利。まず同じ枠で
  勝てるかを見る。

**事前登録した停止条件**（結果を見る前に固定する）:
  1. head-to-head で `decision.verdict` が ACCEPT にならなければ、差し替え案としては
     そこで終了。合成ホールドアウトに枠を使わない。
  2. openai と bert の OOF Spearman が **0.90 未満** かつ bert 単体AP が
     正例率(0.241)の 1.5倍以上（= AP 0.36以上）なら、負けていても
     **PARK: 多様性候補**として記録する。低相関だけを根拠に採用しないこと
     （却下則1「相関が低い＝直交、ではない。信号が無いものは何とも相関しない」）。
  3. 逆に Spearman が 0.90 以上なら、勝っていても「同じ情報の別表現」なので
     差し替えの価値は薄い（exp021 の却下線と同じ）。

  python3 exp/compare_bert_embed.py --emb-model tohoku-bert-v3 --n-seeds 10
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
    CONCAT_EMB_C, CONCAT_EMB_DIM, CONCAT_SLUGS, DEFAULT_MODEL, EMB_LR_PARAMS,
    build_embed_model, load_concat_embeddings, load_embeddings,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
CORR_GATE = 0.90        # これ以上なら「同じ情報の別表現」
AP_FLOOR = 0.36         # 正例率0.241の約1.5倍。多様性候補として記録する最低線


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
            fit = build_embed_model({**params, "C": params["C"][k]}).fit(m[tr], y[tr])
            oof[k][va] = fit.predict_proba(m[va])[:, 1]
    return oof


def build_matrices(emb_model, bert_c, per_col):
    """比較対象の行列を作る。BERT側は非MRLなので dim=None（切り詰めない）。"""
    mats = {
        "openai": load_concat_embeddings(CONCAT_SLUGS, dim=CONCAT_EMB_DIM,
                                         model=DEFAULT_MODEL)[0],
        "bert": load_concat_embeddings(CONCAT_SLUGS, dim=None, model=emb_model)[0],
    }
    cs = {"openai": CONCAT_EMB_C, "bert": bert_c}
    if per_col:
        for slug, key in (("org", "bert_org"), ("overview", "bert_ovw")):
            mats[key] = load_embeddings(slug, dim=None, model=emb_model)[0]
            cs[key] = bert_c
    return mats, cs


def run(emb_model, n_seeds, bert_c, per_col):
    y = pd.read_csv("data/train.csv")["購入フラグ"].values
    mats, cs = build_matrices(emb_model, bert_c, per_col)
    params = {**EMB_LR_PARAMS, "C": cs}

    print(f"行数 {len(y)}  正例率 {y.mean():.4f}  seeds={n_seeds}")
    print(f"  BERT側: {emb_model} (C={bert_c})   "
          f"OpenAI側: {DEFAULT_MODEL} dim={CONCAT_EMB_DIM} C={CONCAT_EMB_C}")
    print("  " + "  ".join(f"{k}:{v.shape[1]}次元" for k, v in mats.items()) + "\n")

    rows = {k: [] for k in mats}
    corrs = []
    for seed in range(n_seeds):
        oof = one_seed(y, mats, params, seed)
        for k in rows:
            rows[k].append(_scores(y, oof[k]))
        corrs.append(spearmanr(oof["openai"], oof["bert"]).statistic)
        print(f"  seed {seed}: "
              + "  ".join(f"{k} AP {rows[k][-1]['ap']:.4f}" for k in rows)
              + f"  r={corrs[-1]:.3f}", flush=True)

    dfs = {k: pd.DataFrame(v) for k, v in rows.items()}
    print("\n=== 平均±std ===")
    summary = pd.DataFrame({k: d.mean() for k, d in dfs.items()}).T
    summary["ap_std"] = [dfs[k]["ap"].std() for k in summary.index]
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    r = float(np.mean(corrs))
    bert_ap = float(dfs["bert"]["ap"].mean())
    print(f"\n=== 相関（事前登録の解釈線 {CORR_GATE}）===")
    print(f"  openai と bert の OOF Spearman: 平均 {r:.4f} "
          f"(最小 {np.min(corrs):.4f} / 最大 {np.max(corrs):.4f})")

    print("\n=== ペア判定（基準 = 現行 openai 連結）===")
    v, d = decision.verdict(dfs["openai"], dfs["bert"])
    print(decision.format_report("H41", f"E4 embedding: openai -> {emb_model}",
                                 dfs["openai"], dfs["bert"], v, d))

    if v.startswith("ACCEPT"):
        nxt = ("差し替え候補。合成ホールドアウトへ: "
               f"python3 exp/holdout_check.py --n-reps 15 --concat-embed --emb-model {emb_model}")
    elif r < CORR_GATE and bert_ap >= AP_FLOOR:
        nxt = (f"PARK(多様性候補): 単体では負けたが r={r:.3f} < {CORR_GATE} かつ "
               f"AP {bert_ap:.4f} >= {AP_FLOOR}。9本目としての追加を合成で測る価値はある")
    else:
        nxt = (f"REJECT: 単体で負け、かつ多様性の条件も未達 "
               f"(r={r:.3f}, AP={bert_ap:.4f})。ここで終了する")
    print(f"\n=== 次の一手 ===\n  {nxt}")

    tag = emb_model.replace("/", "_")
    out = os.path.join(OUT_DIR, f"_h41_bert_{tag}.csv")
    pd.concat([d.assign(model=k, seed=range(len(d))) for k, d in dfs.items()]
              ).to_csv(out, index=False)
    pd.DataFrame([dict(emb_model=emb_model, spearman_openai_bert=r,
                       corr_gate=CORR_GATE, bert_ap=bert_ap, ap_floor=AP_FLOOR,
                       verdict=v, next_step=nxt)]).to_csv(
        os.path.join(OUT_DIR, f"_h41_bert_{tag}_summary.csv"), index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb-model", default="tohoku-bert-v3",
                   help="local_bert_embed.MODELS のキー")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--C", type=float, default=1.0,
                   help="BERT側のLRのC。単体CVで1つに絞ってから固定すること")
    p.add_argument("--per-col", action="store_true",
                   help="bert_org / bert_ovw も並べて、利得の出所を見る")
    a = p.parse_args()
    run(a.emb_model, a.n_seeds, a.C, a.per_col)
