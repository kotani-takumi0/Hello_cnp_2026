"""H93: is there anything left in 企業概要 / 組織図 that E4 has not taken?

The motivation was an EDA fact: PC1 accounts for 95% of what the DX outlook
text can predict, but only 24% (企業概要) and 29% (組織図).  Those two texts are
multi-dimensional, so the reasoning went, there should be room above the
current E4 -- which feeds on both, truncated to 512 dims each
(`CONCAT_SLUGS = ("org", "overview")`).

**That reasoning confuses two things**: multi-dimensional *as text* is not the
same as multi-dimensional *about buying*.  This script measures the second.

Two questions, both against the production blend over 11 seeds:

``add``
    Fit a fresh head on the raw embeddings (full 3072 per column, and the
    512+512 truncation E4 actually uses), add it as a ninth expert, and report
    the standalone AUC, the correlation with the composite, and -- the number
    that decides it -- the **residual AUC** once the composite logit has been
    regressed out.  The bar set in h92 §9.4: residual AUC 0.55 to break even,
    0.60 to matter.  E3_dx_text's own residual AUC is 0.536 for scale.

``replace``
    Swap E4's own predictions for the full-dimension version, keeping eight
    experts.  This is the "raise the quality of one input" form rather than
    the "add an input" form, and it is the one the Matryoshka truncation
    argument would predict something for.

Run:
  .venv/bin/python exp/diagnose_overview_org_ceiling.py --mode add
  .venv/bin/python exp/diagnose_overview_org_ceiling.py --mode replace
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS  # noqa: E402
from ensemble_experts import THS, blend_oof  # noqa: E402

EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
E4_TRUNC = 512          # embedding_features.EMB_DIM。E4 が実際に使う次元
HEAD_C = 0.03
BASE = "base(E4現行)"


def _load(slug):
    z = np.load(f"data/_emb_{slug}_text-embedding-3-large.npz", allow_pickle=True)
    return z["train"].astype(np.float64), z["test"].astype(np.float64)


def _trunc(A, dim):
    """Matryoshka の切り詰め: 先頭dim次元を取って L2 再正規化。"""
    B = A[:, :dim]
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def _representations():
    ov_tr, ov_te = _load("overview")
    og_tr, og_te = _load("org")
    return {
        "企業概要 全3072": (ov_tr, ov_te),
        "組織図 全3072": (og_tr, og_te),
        "概要+組織図 全6144": (np.hstack([ov_tr, og_tr]), np.hstack([ov_te, og_te])),
        f"概要+組織図 {E4_TRUNC}x2(E4と同じ入力)": (
            np.hstack([_trunc(og_tr, E4_TRUNC), _trunc(ov_tr, E4_TRUNC)]),
            np.hstack([_trunc(og_te, E4_TRUNC), _trunc(ov_te, E4_TRUNC)])),
    }


def _crossfit(X_tr, X_te, y, folds, C=HEAD_C):
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = make_pipeline(Normalizer(),
                          LogisticRegression(C=C, max_iter=5000)).fit(X_tr[tr], y[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
    m = make_pipeline(Normalizer(),
                      LogisticRegression(C=C, max_iter=5000)).fit(X_tr, y)
    return oof, m.predict_proba(X_te)[:, 1]


def _score(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS)}


def _lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def _residual_auc(candidate_logit, base_logit, y):
    """合成pの寄与を除いた後に残る順位情報。ここが 0.5 なら独自情報は無い。"""
    fitted = LinearRegression().fit(base_logit.reshape(-1, 1),
                                    candidate_logit).predict(base_logit.reshape(-1, 1))
    return roc_auc_score(y, candidate_logit - fitted)


def run_add(seeds):
    reps = _representations()
    rows = []
    for seed in seeds:
        cache = np.load(CONFIGS["h2h7both4"].format(seed=seed), allow_pickle=True)
        y = cache["y"]
        oof = {n: cache["oof_" + n] for n in EXPERTS}
        te = {n: cache["te_" + n] for n in EXPERTS}
        folds = list(StratifiedKFold(5, shuffle=True,
                                     random_state=seed).split(y.reshape(-1, 1), y))
        base, _, _, _ = blend_oof(y, oof, te, seed=seed, experts=EXPERTS)
        rows.append({"seed": seed, "arm": BASE, **_score(y, base)})
        for name, (A, B) in reps.items():
            o, t = _crossfit(A, B, y, folds)
            cand, _, _, _ = blend_oof(y, {**oof, "E9": o}, {**te, "E9": t},
                                      seed=seed, experts=EXPERTS + ["E9"])
            rows.append({"seed": seed, "arm": name, **_score(y, cand),
                         "solo": roc_auc_score(y, o),
                         "corr": spearmanr(_lg(o), _lg(base)).statistic,
                         "resid": _residual_auc(_lg(o), _lg(base), y)})
    frame = pd.DataFrame(rows)
    b = frame[frame.arm == BASE].sort_values("seed").reset_index(drop=True)
    print("=== 9本目として足す（11seed）===\n")
    print(f"{'表現':<34}{'単体AUC':>8}{'合成と相関':>10}{'残差AUC':>9}"
          f"{'ΔAP':>9}{'ΔAP>0':>7}  判定")
    for name in reps:
        c = frame[frame.arm == name].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{name:<34}{c.solo.mean():>8.3f}{c['corr'].mean():>10.3f}"
              f"{c.resid.mean():>9.3f}{(c.ap - b.ap).mean():>+9.4f}"
              f"{int((c.ap > b.ap).sum()):>5}/11  {v}")
    print("\n  損益分岐は残差AUC 0.55（0.60で明確、h92 §9.4）。E3自身は 0.536。")
    frame.to_csv("exp/overview_org_ceiling.csv", index=False)
    print("saved: exp/overview_org_ceiling.csv")


def run_replace(seeds):
    ov_tr, ov_te = _load("overview")
    og_tr, og_te = _load("org")
    full_tr, full_te = np.hstack([ov_tr, og_tr]), np.hstack([ov_te, og_te])
    rows = []
    for seed in seeds:
        cache = np.load(CONFIGS["h2h7both4"].format(seed=seed), allow_pickle=True)
        y = cache["y"]
        oof = {n: cache["oof_" + n] for n in EXPERTS}
        te = {n: cache["te_" + n] for n in EXPERTS}
        folds = list(StratifiedKFold(5, shuffle=True,
                                     random_state=seed).split(full_tr, y))
        base, _, _, _ = blend_oof(y, oof, te, seed=seed, experts=EXPERTS)
        rows.append({"seed": seed, "arm": BASE, **_score(y, base),
                     "e4solo": roc_auc_score(y, oof["E4_org_text"])})
        for label, C in (("E4を全次元(6144)に差し替え", 0.03),
                         ("E4を全次元・C=0.1に差し替え", 0.1)):
            o, t = _crossfit(full_tr, full_te, y, folds, C=C)
            cand, _, _, _ = blend_oof(y, {**oof, "E4_org_text": o},
                                      {**te, "E4_org_text": t}, seed=seed,
                                      experts=EXPERTS)
            rows.append({"seed": seed, "arm": label, **_score(y, cand),
                         "e4solo": roc_auc_score(y, o)})
    frame = pd.DataFrame(rows)
    b = frame[frame.arm == BASE].sort_values("seed").reset_index(drop=True)
    print("=== E4の入力を差し替える（11seed）===\n")
    print(f"{'構成':<30}{'E4単体AUC':>11}{'ΔAUC':>10}{'ΔAP':>10}{'ΔF1':>10}"
          f"{'ΔAP>0':>7}  判定")
    print(f"{BASE + ' 512x2':<30}{b.e4solo.mean():>11.3f}")
    for label in frame.arm.unique():
        if label == BASE:
            continue
        c = frame[frame.arm == label].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{label:<30}{c.e4solo.mean():>11.3f}{(c.auc - b.auc).mean():>+10.4f}"
              f"{(c.ap - b.ap).mean():>+10.4f}{(c.f1 - b.f1).mean():>+10.4f}"
              f"{int((c.ap > b.ap).sum()):>5}/11  {v}")

    cache = np.load(CONFIGS["h2h7both4"].format(seed=42), allow_pickle=True)
    y = cache["y"]
    o, _ = _crossfit(full_tr, full_te, y,
                     list(StratifiedKFold(5, shuffle=True,
                                          random_state=42).split(full_tr, y)))
    print("\n=== 全次元版はどのExpertと重なっているか（seed42, spearman）===")
    for n in EXPERTS:
        print(f"  {n:<14}{spearmanr(_lg(o), _lg(cache['oof_' + n])).statistic:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="add", choices=("add", "replace"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    if args.mode == "add":
        run_add(args.seeds)
    else:
        run_replace(args.seeds)


if __name__ == "__main__":
    main()
