"""現本命の予測確率の**校正**を診断し、残る誤りの所在を特定する。

問いは、いまの確率 p(x) が真の条件付き確率だったとしたら観測される AUC / AP / F1 は
どうなるか。実測がその分布に収まれば `p` は較正されている。

**重要な限界（2026-08-28に測定）**: このテストは較正しか見ておらず、
「もっと鋭いモデルが存在しないこと」の証拠には**ならない**。較正済みモデルでは
ラベルが周辺的に Bern(p_i) の独立列になるため、真の確率が p でも、より鋭い p* でも、
あらゆる指標の分布が一致してしまう。`--power` で実証できるとおり、logitにノイズを
入れてF1 0.583まで劣化させ較正だけ戻したモデルも |z|<=0.23 で通過する。

したがって本スクリプトから言えるのは次の2点に限られる。

- **決定層（閾値則・F1近似損失・Platt校正）に伸びしろは無い** — このランキングの上では。
- **残る誤りは曖昧帯の数百行に集中している** — これは直接測定であり無条件に成立する。

「このデータの情報上限に達した」とは言えない。より良いランキングの存在は否定できない。

出力:

1. 校正（信頼度図とBrier）
2. 現ランキング上での期待F1（＝決定則の上限。情報の上限ではない）
3. 生成シミュレーション（y ~ Bern(p) を繰り返し、実測をz値で位置づける）
4. 不確実性の所在（曖昧帯の行数、誤りの分布、oracleで解決した場合のF1）
5. `--power` 指定時: 上記テストの検出力の実証

使い方:

    .venv/bin/python exp/diagnose_bayes_ceiling.py
    .venv/bin/python exp/diagnose_bayes_ceiling.py --power

判定に使うのはOOF 742行のみで、testラベルは一切使わない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_experts import blend_oof  # noqa: E402

CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
THS = np.linspace(0.05, 0.95, 181)


def load_blend(cache, seed):
    z = np.load(cache, allow_pickle=True)
    y = z["y"]
    oof = {n: z["oof_" + n] for n in EXPERTS}
    te = {n: z["te_" + n] for n in EXPERTS}
    p, pte, _, _ = blend_oof(y, oof, te, seed=seed, alpha=None, experts=EXPERTS)
    return y, np.clip(p, 1e-6, 1 - 1e-6), np.clip(pte, 1e-6, 1 - 1e-6)


def best_f1_fast(y, order, n_pos=None):
    """降順に並べた確率に対する top-k 走査で最大F1を返す。"""
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    P = ys.sum() if n_pos is None else n_pos
    if P == 0:
        return 0.0
    return float((2 * tp / (k + P)).max())


def fast_auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def section_calibration(y, p):
    print("=" * 72)
    print("1. 校正 — 予測確率はそのまま購入確率として読めるか")
    print("=" * 72)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(p, edges) - 1, 0, 9)
    print(f"{'bin':>4} {'n':>5} {'mean_p':>8} {'observed':>9} {'diff':>8}")
    for i in range(10):
        m = b == i
        print(f"{i:>4} {m.sum():>5} {p[m].mean():>8.3f} {y[m].mean():>9.3f} "
              f"{y[m].mean() - p[m].mean():>+8.3f}")
    cal = np.zeros_like(p)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(p.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        cal[va] = iso.predict(p[va])
    print(f"\nBrier 生 {np.mean((p - y) ** 2):.4f} / isotonic-CV {np.mean((cal - y) ** 2):.4f}"
          "  （isotonicで下がらない = 校正済み）")
    return cal


def section_power(y, p, n_sim, seed):
    """このテストの検出力を実証する。

    logitにノイズを足してランキングを壊し、isotonicで較正だけ戻したモデルを
    同じテストにかける。実力が落ちているのに通過するなら、テストは
    「もっと良いモデルの不在」を主張する根拠にならない。
    """
    print()
    print("=" * 72)
    print("5. 検出力の実証 — わざと劣化させた較正済みモデルは見破れるか")
    print("=" * 72)
    order = np.argsort(-p)
    print(f"  {'モデル':<28}{'実AUC':>9}{'実最大F1':>10}   z値 (auc / ap / f1)")

    def one(score, tag):
        s = np.clip(score, 1e-6, 1 - 1e-6)
        o = np.argsort(-s)
        rng = np.random.default_rng(seed)
        R = {"auc": [], "ap": [], "f1": []}
        for _ in range(n_sim):
            yy = (rng.random(len(s)) < s).astype(int)
            if yy.sum() < 5:
                continue
            R["auc"].append(fast_auc(yy, s))
            R["ap"].append(average_precision_score(yy, s))
            R["f1"].append(best_f1_fast(yy, o))
        obs = dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                   f1=best_f1_fast(y, o))
        zs = {k: (obs[k] - np.mean(v)) / np.std(v) for k, v in R.items()}
        print(f"  {tag:<28}{obs['auc']:>9.4f}{obs['f1']:>10.4f}   "
              f"{zs['auc']:+.2f} / {zs['ap']:+.2f} / {zs['f1']:+.2f}")

    one(p, "本命 p")
    rng = np.random.default_rng(7)
    for noise in [0.5, 1.0, 2.0, 4.0]:
        deg = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + rng.normal(0, noise, len(p)))))
        iso = IsotonicRegression(out_of_bounds="clip").fit(deg, y)
        one(np.clip(iso.predict(deg), 1e-6, 1 - 1e-6), f"劣化(logitノイズ sd={noise})")
    print("\n  → 劣化モデルも同じか小さい|z|で通過する。このテストの検出力はゼロに近い。")
    print("     結論は『較正されている』までであり、『上限に達した』ではない。")
    _ = order


def section_ceiling(y, p, cal):
    print()
    print("=" * 72)
    print("2. 現ランキング上での期待F1（決定則の上限。情報の上限ではない）")
    print("=" * 72)
    for tag, prob in [("生の合成確率", p), ("isotonic-CV後", cal)]:
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        order = np.argsort(-prob)
        ps = prob[order]
        k = np.arange(1, len(prob) + 1)
        ef1 = 2 * np.cumsum(ps) / (k + prob.sum())
        kstar = int(np.argmax(ef1)) + 1
        print(f"  {tag:14s} 最適top-k = {kstar:3d} ({kstar / len(y) * 100:.1f}%)  "
              f"期待F1* = {ef1.max():.4f}")
    achieved = max(f1_score(y, (p >= t).astype(int)) for t in THS)
    half = achieved / 2
    print(f"\n  実測 OOF F1 (oof_best) = {achieved:.4f}")
    print(f"  実測 OOF F1 (half_f1)  = {f1_score(y, (p >= half).astype(int)):.4f}")
    print("\n  → 実測が期待F1*と並ぶ = このランキングを固定した上での決定則は最適。")
    print("     ランキング自体を良くすればこの数字も上がる（--power 参照）。")


def section_simulation(y, p, n_sim, seed):
    print()
    print("=" * 72)
    print(f"3. 生成シミュレーション — y ~ Bern(p) を{n_sim}回引いて実測を位置づける")
    print("=" * 72)
    band = (p > 0.1) & (p < 0.9)
    order = np.argsort(-p)
    rng = np.random.default_rng(seed)
    keys = ["auc", "ap", "f1", "band_auc", "n_pos"]
    R = {k: [] for k in keys}
    for _ in range(n_sim):
        yy = (rng.random(len(p)) < p).astype(int)
        if yy.sum() < 5:
            continue
        R["auc"].append(fast_auc(yy, p))
        R["ap"].append(average_precision_score(yy, p))
        R["f1"].append(best_f1_fast(yy, order))
        R["n_pos"].append(yy.sum())
        if 0 < yy[band].sum() < band.sum():
            R["band_auc"].append(fast_auc(yy[band], p[band]))
    obs = dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
               f1=best_f1_fast(y, order), band_auc=roc_auc_score(y[band], p[band]),
               n_pos=float(y.sum()))
    print(f"{'指標':<10}{'実測':>10}{'シミュ平均':>12}{'シミュsd':>10}{'z':>8}")
    zs = []
    for k in keys:
        v = np.asarray(R[k], dtype=float)
        z = (obs[k] - v.mean()) / v.std()
        zs.append(z)
        print(f"{k:<10}{obs[k]:>10.4f}{v.mean():>12.4f}{v.std():>10.4f}{z:>+8.2f}")
    print(f"\n  平均|z| = {np.mean(np.abs(zs)):.2f}  （すべて|z|<2なら「pは較正されている」）")
    print("  注意: これは較正の確認であって、より鋭いモデルの不在の証拠ではない。--power を参照。")


def section_uncertainty(y, p, pte):
    print()
    print("=" * 72)
    print("4. 不確実性の所在 — 残った誤りはどの行にあるか")
    print("=" * 72)
    th = max(f1_score(y, (p >= t).astype(int)) for t in THS) / 2
    pred = (p >= th).astype(int)
    err = pred != y
    print(f"OOF誤り {err.sum()}件 (FP {int(((pred == 1) & (y == 0)).sum())} / "
          f"FN {int(((pred == 0) & (y == 1)).sum())})、閾値 {th:.4f}\n")
    print(f"{'確率帯':<10}{'train行数':>10}{'誤り':>7}{'誤りの割合':>12}{'test行数':>10}")
    for lo, hi, lab in [(0, .1, "p<0.1"), (.1, .3, "0.1-0.3"), (.3, .5, "0.3-0.5"),
                        (.5, .7, "0.5-0.7"), (.7, .9, "0.7-0.9"), (.9, 1.01, "p>0.9")]:
        m = (p >= lo) & (p < hi)
        mte = (pte >= lo) & (pte < hi)
        print(f"{lab:<10}{m.sum():>10}{int(err[m].sum()):>7}"
              f"{100 * err[m].sum() / max(err.sum(), 1):>11.1f}%{mte.sum():>10}")

    print("\n曖昧帯の広さ（testで実際に勝負が決まる行数）")
    for lo, hi in [(.02, .98), (.05, .95), (.10, .90), (.15, .85), (.20, .80)]:
        conf_tr = (p < lo) | (p > hi)
        lab = (p[conf_tr] > 0.5).astype(int)
        print(f"  {lo:.2f}<p<{hi:.2f}: 曖昧 train {int(((p >= lo) & (p <= hi)).sum()):3d} / "
              f"test {int(((pte >= lo) & (pte <= hi)).sum()):3d}  ｜確信帯の擬似ラベル誤り率 "
              f"{(lab != y[conf_tr]).mean() * 100:4.1f}% (test {int(conf_tr.sum() and ((pte < lo) | (pte > hi)).sum()):3d}行が利用可)")

    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    o = np.argsort(-ent)
    print(f"\nエントロピー総量 {ent.sum():.1f} bits / 742行。oracleで上位k行を解決した場合のF1:")
    base = f1_score(y, pred)
    for k in [10, 20, 30, 50, 75, 100, 150]:
        q = pred.copy()
        q[o[:k]] = y[o[:k]]
        print(f"  k={k:3d} ({100 * ent[o[:k]].sum() / ent.sum():4.1f}%のエントロピー) "
              f"→ F1 {f1_score(y, q):.4f} (+{f1_score(y, q) - base:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sim", type=int, default=600)
    ap.add_argument("--power", action="store_true",
                    help="このテストの検出力を実証する（劣化モデルも通過することを示す）")
    args = ap.parse_args()

    y, p, pte = load_blend(args.cache, args.seed)
    cal = section_calibration(y, p)
    section_ceiling(y, p, cal)
    section_simulation(y, p, args.n_sim, args.seed)
    section_uncertainty(y, p, pte)
    if args.power:
        section_power(y, p, min(args.n_sim, 400), args.seed)


if __name__ == "__main__":
    main()
