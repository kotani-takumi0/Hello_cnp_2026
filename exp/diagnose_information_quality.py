"""H94: where is there room to raise the *quality* of the information itself?

The one quality upgrade that ever worked here was E0's: delete 29 genuinely
noisy hand-made columns, replace them with a better instrument (the LLM
4-axis extraction).  ΔAP +0.0198.  The same operation on E1/E2/E6 was neutral
or worse (H52/H54/H56/H57).  So the question is not "should we prune" but
**where are the two preconditions still met** -- noise worth deleting, and a
better instrument to replace it with.

Three measurements, all label-free except the final scoring:

``richness``
    Is the ambiguous band a set of genuinely middling companies, or a set of
    rows whose *measurements* degenerated?  Compares, band vs confident, the
    richness of every source: survey response spread, count of neutral 3s,
    text length, distance from the corpus centroid, org-chart size, missing
    counts.  A band made of failed measurements would be a quality target;
    a band made of middling companies is not.

``survey``
    The survey is the only expert whose information content demonstrably
    varies by row: E2's AUC is ~0.11 higher for respondents who differentiate
    across items than for those who answer flat.  This runs the terciles over
    11 seeds, then the decisive test -- let the meta change E2's weight per
    row using that spread -- on the production folds.

``independence``
    Measurement-error correction needs a second indicator of the same
    construct.  This asks whether *anything* in the dataset can reconstruct
    the survey answers: text embeddings (ridge), finance+scale (LightGBM),
    both, and the reverse direction.  R^2 <= 0 everywhere means no second
    indicator exists and the survey cannot be denoised from within the data.

Run:
  .venv/bin/python exp/diagnose_information_quality.py --mode richness
  .venv/bin/python exp/diagnose_information_quality.py --mode survey
  .venv/bin/python exp/diagnose_information_quality.py --mode independence
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_short_term_debt_gate import _blend_oof_p  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS  # noqa: E402
from expert_groups import FINANCE_COLS, SCALE_COLS  # noqa: E402
from meta_blend import fit_meta, select_alpha  # noqa: E402

EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
# 常に観測される5段階9項目。a6は2値、a7は条件付き欠損なので分布統計から外す。
FIVE = ("アンケート１", "アンケート２", "アンケート３", "アンケート４", "アンケート５",
        "アンケート８", "アンケート９", "アンケート１０", "アンケート１１")


def _emb(slug, n=None):
    z = np.load(f"data/_emb_{slug}_text-embedding-3-large.npz", allow_pickle=True)
    return z["train"].astype(np.float64)


def _survey_spread(raw):
    return raw[list(FIVE)].astype(float).std(axis=1).values


def run_richness(seed, config):
    raw = pd.read_csv("data/train.csv")
    y, Z, _ = _load(CONFIGS[config], seed)
    p = _blend_oof_p(y, Z, seed, "auto")
    group = np.where(p < 0.1, "確信-非購入",
                     np.where(p > 0.9, "確信-購入", "曖昧帯"))
    S = raw[list(FIVE)].astype(float)
    dx = _emb("dx_outlook")
    centre = dx.mean(0)
    centre /= np.linalg.norm(centre)
    pc1 = PCA(n_components=1, random_state=0).fit(dx).transform(dx)[:, 0]
    unit = dx / np.linalg.norm(dx, axis=1, keepdims=True)

    R = pd.DataFrame({
        "アンケ_項目間SD": S.std(axis=1).values,
        "アンケ_3の個数": (S == 3).sum(axis=1).values,
        "アンケ_極端回答数": ((S == 1) | (S == 5)).sum(axis=1).values,
        "Q7欠損": raw["アンケート７"].isna().astype(int).values,
        "DX展望_文字数": raw["今後のDX展望"].fillna("").str.len().values,
        "DX展望_PC1絶対値": np.abs(pc1),
        "DX展望_重心距離": 1 - unit @ centre,
        "企業概要_文字数": raw["企業概要"].fillna("").str.len().values,
        "組織図_部署数": raw["組織図"].fillna("").astype(str).str.count("─").values,
        "規模列_欠損数": raw[["事業所数", "工場数", "店舗数", "資本金",
                             "営業利益", "経常利益"]].isna().sum(axis=1).values,
    })
    print("=== 帯は『中庸な企業』か『退化した計測』か（群別平均）===\n")
    t = R.groupby(group).mean().T
    t["帯/確信"] = t["曖昧帯"] / ((t["確信-非購入"] + t["確信-購入"]) / 2)
    print(t[["確信-非購入", "曖昧帯", "確信-購入", "帯/確信"]].round(3).to_string())
    print("\n  比が 1.00 なら、帯の行の計測は確信群と同じだけ豊か"
          "＝『測れなかった行』ではない。")

    band = (p > 0.1) & (p < 0.9)
    correct = ((p > 0.5).astype(int) == y).astype(int)
    print(f"\n=== 帯の中で、計測の豊かさは正誤を分けるか"
          f"（帯{int(band.sum())}行、正解率 {correct[band].mean():.3f}）===\n")
    print(f"{'指標':<22}{'正解行':>10}{'誤り行':>10}{'AUC(正誤)':>12}")
    for c in R.columns:
        v = R[c].values[band]
        cb = correct[band]
        print(f"{c:<22}{v[cb == 1].mean():>10.2f}{v[cb == 0].mean():>10.2f}"
              f"{roc_auc_score(cb, v):>12.3f}")
    print("\n  ※単一seedの表。目立つ値が出ても --mode survey と同じく"
          "11seedで確認すること（組織図_部署数は seed で 0.48〜0.59 に振れる）。")


def _tercile(spread):
    return pd.qcut(spread, 3, labels=[0, 1, 2]).to_numpy()


def _meta_oof(Z, y, seed, extra=None):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(Z, y):
        M = Z if extra is None else np.column_stack([Z, extra])
        alpha = select_alpha(M[tr], y[tr], seed=seed)[0]
        w, b = fit_meta(M[tr], y[tr], alpha=alpha)
        p[va] = expit(M[va] @ w + b)
    return p


def _score(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS)}


def run_survey(seeds, config):
    raw = pd.read_csv("data/train.csv")
    spread = _survey_spread(raw)
    tercile = _tercile(spread)

    print("=== 1. E2の情報量は回答分散で変わるか（3分位、seedごと）===\n")
    print(f"{'seed':>5}{'退化(SD小)':>11}{'中':>9}{'豊か(SD大)':>11}{'豊か-退化':>10}")
    gaps = []
    for seed in seeds:
        c = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
        y, e2 = c["y"], c["oof_E2_survey"]
        a = [roc_auc_score(y[tercile == k], e2[tercile == k]) for k in (0, 1, 2)]
        gaps.append(a[2] - a[0])
        print(f"{seed:>5}{a[0]:>11.3f}{a[1]:>9.3f}{a[2]:>11.3f}{a[2] - a[0]:>+10.3f}")
    print(f"{'平均':>5}{'':>31}{np.mean(gaps):>+10.3f}"
          f"   正の回数 {sum(1 for g in gaps if g > 0)}/{len(gaps)}")

    print("\n=== 2. 決定的テスト: 行ごとにE2の重みを変えてよいことにする ===\n")
    rows = []
    for seed in seeds:
        c = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
        y = c["y"]
        Z = logit(np.clip(np.column_stack([c["oof_" + n] for n in EXPERTS]),
                          1e-6, 1 - 1e-6))
        s = (spread - spread.mean()) / spread.std()
        l2 = Z[:, EXPERTS.index("E2_survey")]
        l0 = Z[:, EXPERTS.index("E0_anchor")]
        rows.append({"seed": seed, "arm": "base", **_score(y, _meta_oof(Z, y, seed))})
        # メタの重みは非負なので、交互作用は ±両方の列を渡して符号を自由にする。
        for name, extra in (
                ("E2×回答分散", np.column_stack([l2 * s, -l2 * s])),
                ("E2×分散 + E0×分散",
                 np.column_stack([l2 * s, -l2 * s, l0 * s, -l0 * s])),
                ("分散を単独で足す", np.column_stack([s, -s]))):
            rows.append({"seed": seed, "arm": name,
                         **_score(y, _meta_oof(Z, y, seed, extra))})
    r = pd.DataFrame(rows)
    b = r[r.arm == "base"].sort_values("seed").reset_index(drop=True)
    print(f"{'構成':<22}{'ΔAUC':>10}{'ΔAP':>10}{'ΔF1':>10}{'ΔAP>0':>8}  判定")
    for name in ("E2×回答分散", "E2×分散 + E0×分散", "分散を単独で足す"):
        cand = r[r.arm == name].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, cand)
        print(f"{name:<22}{(cand.auc - b.auc).mean():>+10.4f}"
              f"{(cand.ap - b.ap).mean():>+10.4f}{(cand.f1 - b.f1).mean():>+10.4f}"
              f"{int((cand.ap > b.ap).sum()):>6}/{len(cand)}  {v}")
    r.to_csv("exp/information_quality_survey.csv", index=False)
    print("\nsaved: exp/information_quality_survey.csv")


def _r2_cv(X, target, tree=False):
    o = np.zeros(len(target))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        if tree:
            m = lgb.train(dict(objective="regression", learning_rate=0.05,
                               num_leaves=8, verbosity=-1, num_threads=8, seed=0),
                          lgb.Dataset(X[a], target[a]), num_boost_round=200)
            o[b] = m.predict(X[b])
        else:
            o[b] = Ridge(alpha=100.0).fit(X[a], target[a]).predict(X[b])
    return 1 - ((target - o) ** 2).sum() / ((target - target.mean()) ** 2).sum()


def run_independence():
    raw = pd.read_csv("data/train.csv")
    S = raw[list(FIVE)].astype(float).values
    fin = raw[list(FINANCE_COLS) + list(SCALE_COLS)].astype(float).values
    txt = np.hstack([_emb("dx_outlook"), _emb("overview"), _emb("org")])
    both = np.hstack([fin, txt[:, :200]])

    print("=== アンケート9項目は、他の情報から再現できるか（CV R2、ラベル不使用）===\n")
    print(f"{'項目':<12}{'テキスト(リッジ)':>16}{'財務+規模(LGBM)':>17}{'両方(LGBM)':>13}")
    for j, c in enumerate(FIVE):
        print(f"{c:<12}{_r2_cv(txt, S[:, j]):>16.3f}"
              f"{_r2_cv(fin, S[:, j], tree=True):>17.3f}"
              f"{_r2_cv(both, S[:, j], tree=True):>13.3f}")
    print("\n  R2<=0 は『まったく再現できない』。")
    print("  測定誤差の補正には同じ構成概念の第二の指標が要るが、"
          "それがデータ内に存在しないということ。")

    print("\n=== 逆向き: アンケートから他を再現できるか ===")
    for label, target in (("売上(log)", np.log1p(raw["売上"].astype(float).values)),
                          ("従業員数(log)", np.log1p(raw["従業員数"].astype(float).values))):
        print(f"  {label:<14} R2 {_r2_cv(S, target, tree=True):>6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="richness",
                    choices=("richness", "survey", "independence"))
    ap.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    if args.mode == "richness":
        run_richness(args.seed, args.config)
    elif args.mode == "survey":
        run_survey(args.seeds, args.config)
    else:
        run_independence()


if __name__ == "__main__":
    main()
