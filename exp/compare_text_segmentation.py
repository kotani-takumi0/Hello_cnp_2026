"""テキストを細かく割ると強くなるか（H64）。

3つの問いを順に測る。

1. **分割**: `今後のDX展望` を位置で割ったブロック単体、および文単位MIL集約は、
   全文一括を超えるか。
2. **併用**: 全文スコアに末尾スコアを足した2列は、全文単体を超えるか。
3. **そもそもの上限**: テキスト系Expert(E3/E4)を現本命から抜くと何が失われるか。
   ここが小さいなら、テキストの読み方をどれだけ改良しても最終スコアは動かない。

H10(char n-gram設定)とH12(時制分割)の続きにあたる。あちらは分割の**粒度**と
**時制**を変えたが、文単位のMIL集約と位置ブロックは未実施だった。

使い方:

    .venv/bin/python exp/compare_text_segmentation.py
    .venv/bin/python exp/compare_text_segmentation.py --seeds 42 0 1 2 3 --skip-mil
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_experts import blend_oof  # noqa: E402

TFIDF = dict(analyzer="char_wb", ngram_range=(1, 3), min_df=5, sublinear_tf=True)
LR = dict(C=1.0, solver="liblinear", dual=True, max_iter=3000, random_state=0)
CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]


def split_sents(text):
    s = [x.strip() for x in re.split(r"(?<=。)", text.replace("\n", "")) if len(x.strip()) >= 10]
    return s or [text]


def segment(sents, mode):
    n = len(sents)
    if mode == "full":
        return "".join(sents)
    if mode == "head_half":
        return "".join(sents[:int(n * 0.5)])
    if mode == "last_half":
        return "".join(sents[int(n * 0.5):])
    if mode == "last_third":
        return "".join(sents[int(n * 2 / 3):])
    if mode == "last_quarter":
        return "".join(sents[int(n * 0.75):])
    if mode.startswith("last") and mode[4:].isdigit():
        return "".join(sents[-int(mode[4:]):])
    raise ValueError(mode)


def tfidf_oof(texts, y, seed):
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        A = [texts[i] for i in a]
        B = [texts[i] for i in b]
        v = TfidfVectorizer(**TFIDF).fit(A)
        m = LogisticRegression(**LR).fit(v.transform(A), y[a])
        out[b] = m.predict_proba(v.transform(B))[:, 1]
    return out


def mil_oof(sents_all, y, agg, seed, topk=3):
    """文をそれぞれ独立に採点し、文書レベルへ集約する（naive MIL）。"""
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        Xs, ys = [], []
        for i in a:
            for s in sents_all[i]:
                Xs.append(s)
                ys.append(y[i])
        v = TfidfVectorizer(**TFIDF).fit(Xs)
        m = LogisticRegression(**LR).fit(v.transform(Xs), np.asarray(ys))
        for i in b:
            sc = m.decision_function(v.transform(sents_all[i]))
            if agg == "max":
                out[i] = sc.max()
            elif agg == "mean":
                out[i] = sc.mean()
            elif agg == "topk":
                out[i] = np.sort(sc)[-min(topk, len(sc)):].mean()
            elif agg == "max_min":
                out[i] = sc.max() + sc.min()
    return out


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap_.add_argument("--skip-mil", action="store_true")
    args = ap_.parse_args()

    tr = pd.read_csv("data/train.csv")
    y = tr["購入フラグ"].values
    docs = tr["今後のDX展望"].astype(str).tolist()
    sents_all = [split_sents(d) for d in docs]
    print(f"文数: 平均 {np.mean([len(s) for s in sents_all]):.1f} / 合計 {sum(len(s) for s in sents_all)}")

    modes = ["full", "head_half", "last_half", "last_third", "last_quarter", "last3", "last2"]
    store = {}
    scores = {m: {"auc": [], "ap": []} for m in modes}
    for seed in args.seeds:
        for m in modes:
            o = tfidf_oof([segment(s, m) for s in sents_all], y, seed)
            store[(m, seed)] = o
            scores[m]["auc"].append(roc_auc_score(y, o))
            scores[m]["ap"].append(average_precision_score(y, o))

    print(f"\n=== 1. セグメント単体（{len(args.seeds)}seed平均）===")
    print(f"{'セグメント':<16}{'AUC':>8}{'AP':>8}{'vs full ΔAP':>14}  勝率")
    for m in modes:
        d = np.array(scores[m]["ap"]) - np.array(scores["full"]["ap"])
        print(f"{m:<16}{np.mean(scores[m]['auc']):>8.4f}{np.mean(scores[m]['ap']):>8.4f}"
              f"{d.mean():>+14.4f}  {int((d > 0).sum())}/{len(args.seeds)}")

    if not args.skip_mil:
        print(f"\n=== 2. 文単位MIL集約（{len(args.seeds)}seed平均）===")
        for agg in ["mean", "max", "topk", "max_min"]:
            au, apv = [], []
            for seed in args.seeds:
                o = mil_oof(sents_all, y, agg, seed)
                au.append(roc_auc_score(y, o))
                apv.append(average_precision_score(y, o))
            d = np.array(apv) - np.array(scores["full"]["ap"])
            print(f"  MIL {agg:<10} AUC {np.mean(au):.4f}  AP {np.mean(apv):.4f}"
                  f"   vs full ΔAP {d.mean():+.4f} ({int((d > 0).sum())}/{len(args.seeds)})")

    print(f"\n=== 3. 全文 + 末尾 の2列スタック（{len(args.seeds)}seed平均）===")
    for m in ["last_quarter", "last_third", "last3", "last2"]:
        au, apv = [], []
        for seed in args.seeds:
            Z = np.column_stack([logit(store[("full", seed)]), logit(store[(m, seed)])])
            o = np.zeros(len(y))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed + 100).split(Z, y):
                mm = LogisticRegression(max_iter=3000, random_state=0).fit(Z[a], y[a])
                o[b] = mm.predict_proba(Z[b])[:, 1]
            au.append(roc_auc_score(y, o))
            apv.append(average_precision_score(y, o))
        d = np.array(apv) - np.array(scores["full"]["ap"])
        print(f"  full + {m:<14} AUC {np.mean(au):.4f}  AP {np.mean(apv):.4f}"
              f"   vs full ΔAP {d.mean():+.4f} ({int((d > 0).sum())}/{len(args.seeds)})")

    print("\n=== 4. そもそもテキストExpertは最終合成へ何を足しているか ===")
    z = np.load(CACHE, allow_pickle=True)
    oof = {n: z["oof_" + n] for n in EXPERTS}
    te = {n: z["te_" + n] for n in EXPERTS}
    p8, _, _, _ = blend_oof(y, oof, te, seed=42, alpha=None, experts=EXPERTS)
    band = (p8 > 0.1) & (p8 < 0.9)

    def ev(ex):
        au, apv, ba = [], [], []
        for s in args.seeds:
            q, _, _, _ = blend_oof(y, oof, te, seed=s, alpha=None, experts=ex)
            au.append(roc_auc_score(y, q))
            apv.append(average_precision_score(y, q))
            ba.append(roc_auc_score(y[band], q[band]))
        return np.array(au), np.array(apv), np.array(ba)

    cfg = {
        "8本（現本命）": EXPERTS,
        "テキスト2本を除く": [n for n in EXPERTS if n not in ("E3_dx_text", "E4_org_text")],
        "E3のみ除く": [n for n in EXPERTS if n != "E3_dx_text"],
        "E4のみ除く": [n for n in EXPERTS if n != "E4_org_text"],
        "E0+E0b のみ": ["E0_anchor", "E0b_linear"],
    }
    base = None
    print(f"{'構成':<24}{'AUC':>8}{'AP':>8}{'曖昧帯AUC':>11}   vs 8本 ΔAP")
    for tag, ex in cfg.items():
        au, apv, ba = ev(ex)
        if base is None:
            base = apv
        d = apv - base
        print(f"{tag:<24}{au.mean():>8.4f}{apv.mean():>8.4f}{ba.mean():>11.4f}"
              f"   {d.mean():+.4f} ({int((d > 0).sum())}/{len(args.seeds)})")


if __name__ == "__main__":
    main()
