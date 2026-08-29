"""曖昧帯（現本命が迷う行）のプロファイルと、ドメイン仮説特徴の帯内スクリーニング。

現本命の合成OOF確率 `p` で 3群に分ける。

    確信-非購入 (p<0.1) / 曖昧帯 (0.1<=p<=0.9) / 確信-購入 (p>0.9)

確信2群は誤りをほとんど出さないので、F1は曖昧帯だけで決まる。
そこで「誰が帯に落ちるか」と「帯の中で何が購入を分けるか」を分けて測る。

出力:

1. 群別の件数・実購入率
2. 誰が帯に落ちるか（業界・上場種別・特徴・数値プロファイル）
3. 帯の中で購入/非購入を分ける生列（ブートストラップ95%CI付き）
4. 帯の中での各Expertの識別力
5. ドメイン仮説から作った特徴の帯内スクリーニング（多重補正付き）
6. **決定的テスト**: 現本命スコアの上にそれを足すと帯内AUC/APが動くか

5でCIが0.5を跨がなくても、6で動かなければ採用しない。過去の
exp020/023/025 と同型の罠（単体で強い != 合成で効く）を避けるため。

使い方:

    .venv/bin/python exp/profile_uncertain_band.py
    .venv/bin/python exp/profile_uncertain_band.py --lo 0.15 --hi 0.85
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_experts import blend_oof  # noqa: E402

CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
# E7 の不満スコアが使う4問。Q8 は入っていない。
FUMAN4 = ("アンケート７", "アンケート１０", "アンケート２", "アンケート４")
UNINSTALLED_A7 = 3.3


def safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def domain_features(d):
    """ドメイン仮説から作る候補。すべて既存42列の変換であることに注意。"""
    emp = d["従業員数"]
    soft = -d["無形固定資産変動(ソフトウェア関連)"]      # アウトフローを正値へ
    a7 = d["アンケート７"].fillna(UNINSTALLED_A7)
    org = d["組織図"].astype(str)
    q8gap = 6 - d["アンケート８"]                       # 成果を実感できていない度合い
    sites = d["事業所数"].fillna(0) + d["工場数"].fillna(0) + d["店舗数"].fillna(0)
    field_ratio = safe_div(d["工場数"].fillna(0) + d["店舗数"].fillna(0), sites + 1)

    F = {}
    # 仮説1: 教育予算のオーナー（買い手）が組織図に居るか
    F["組織図_人事系部門あり"] = org.str.contains("人事|人材|人財|教育|研修|能力開発|タレント").astype(int)
    F["組織図_人材開発専門部署"] = org.str.contains("人材開発|人財開発|教育|研修|能力開発").astype(int)
    F["組織図_部署数"] = org.str.count("─").astype(float)
    # 仮説2: 研修対象人口（現場比率で割り引いた知識労働者数）
    F["現場拠点比率"] = field_ratio
    F["推定知識労働者数_log"] = np.log1p(emp * (1 - field_ratio))
    F["1拠点あたり従業員"] = safe_div(emp, sites + 1)
    # 仮説3: 席課金なので総額でなく一人あたり支払い能力
    F["営業利益_per_従業員"] = safe_div(d["営業利益"], emp)
    F["売上_per_従業員"] = safe_div(d["売上"], emp)
    F["営業CF_per_従業員"] = safe_div(d["営業CF"], emp)
    F["ソフト投資_per_従業員"] = safe_div(soft, emp)
    # 仮説4: ツール先行・人材後追いギャップ（道具は買ったが使えていない）
    F["ソフト投資per人 x 成果未実感"] = F["ソフト投資_per_従業員"] * q8gap
    F["ソフト投資売上比 x 成果未実感"] = safe_div(soft, d["売上"]) * q8gap
    F["log_ソフト投資 x 成果未実感"] = np.log1p(soft) * q8gap
    F["ソフト投資per人 x ツール不満"] = F["ソフト投資_per_従業員"] * (6 - a7)
    # 仮説5: 技術は入ったが人が追いついていない
    F["Q3導入 - Q8成果"] = d["アンケート３"] - d["アンケート８"]
    F["Q5セキュリティ - Q8成果"] = d["アンケート５"] - d["アンケート８"]
    F["Q1戦略 - Q8成果"] = d["アンケート１"] - d["アンケート８"]
    F["Q11情報収集 - Q3導入"] = d["アンケート１１"] - d["アンケート３"]
    F["Q1戦略 x Q8成果未実感"] = d["アンケート１"] * q8gap
    # 仮説6: 資本集約 vs 人材集約
    F["減価償却費_総資産比"] = safe_div(d["減価償却費"], d["総資産"])
    F["減価償却費_per_従業員"] = safe_div(d["減価償却費"], emp)
    F["有形投資_ソフト投資比"] = safe_div(-d["有形固定資産変動"], soft + 1)
    # 仮説7: E7の不満スコアに Q8 を足す
    f4 = sum((6 - (a7 if c == "アンケート７" else d[c])) for c in FUMAN4)
    F["不満4(E7現行)"] = f4
    F["不満5(Q7,10,2,4,8)"] = f4 + q8gap
    F["不満_Q8のみ"] = q8gap
    return F


# 「教育商材の購買直前の企業が書きそうなこと」をキーワードで代理する。
# 42列の外＝テキストにしか無い情報が帯で効くかを見るため。
TEXT_MARKERS = {
    "人材育成の明示": "人材育成|人財育成|人材開発|育成",
    "リスキリング/スキル": "リスキリング|リスキル|スキル|デジタル人材",
    "研修/教育の明示": "研修|教育|トレーニング|e-?ラーニング|eラーニング",
    "リテラシー": "リテラシー",
    "人材不足の訴え": "人材不足|人手不足|スキル不足|要員不足|担い手",
    "内製化志向": "内製|自前|社内で開発|自社開発",
    "外部委託/ベンダー": "ベンダー|外部委託|パートナー|アウトソース|委託",
    "予算/投資額の明示": "予算|投資額|億円|百万円|投資枠",
    "段階的/PoC": "段階的|パイロット|PoC|POC|実証",
    "経営層主導": "経営層|トップダウン|経営主導|社長|取締役",
    "全社展開": "全社|全部門|横展開",
    "数値目標/KPI": "KPI|目標値|％|%|20[0-9][0-9]年",
    "成果が出ていない": "限定的|とどまり|進んでいない|課題が残|不十分|遅れ",
    "既に十分やっている": "先進|屈指|いち早く|高水準|業界を?リード|先行",
    "教育投資の明示": "DX教育|教育投資|人材投資|育成投資",
}


def band_auc_ci(v, yb, band, rng, n_boot=2000):
    v = np.asarray(v, dtype=float)[band]
    m = ~np.isnan(v)
    if m.sum() < 60 or not (0 < yb[m].sum() < m.sum()):
        return None
    a = roc_auc_score(yb[m], v[m])
    bs = []
    for _ in range(n_boot):
        i = rng.choice(m.sum(), m.sum(), replace=True)
        if 0 < yb[m][i].sum() < len(i):
            bs.append(roc_auc_score(yb[m][i], v[m][i]))
    q = np.quantile(bs, [.0025, .025, .975, .9975])
    return a, q[1], q[2], m.sum(), q[0], q[3]


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--lo", type=float, default=0.1)
    ap_.add_argument("--hi", type=float, default=0.9)
    ap_.add_argument("--seed", type=int, default=42)
    args = ap_.parse_args()

    z = np.load(CACHE, allow_pickle=True)
    y = z["y"]
    oof = {n: z["oof_" + n] for n in EXPERTS}
    te = {n: z["te_" + n] for n in EXPERTS}
    p, pte, _, _ = blend_oof(y, oof, te, seed=args.seed, alpha=None, experts=EXPERTS)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    band = (p > args.lo) & (p < args.hi)
    yb = y[band]
    d = pd.read_csv("data/train.csv")
    d["_g"] = pd.cut(p, [0, args.lo, args.hi, 1.0],
                     labels=["確信-非購入", "曖昧帯", "確信-購入"])

    print("=" * 78)
    print(f"1. 群の定義（閾値 {args.lo}/{args.hi}）")
    print("=" * 78)
    print(d.groupby("_g").agg(n=("購入フラグ", "size"), 実購入率=("購入フラグ", "mean")).round(3).to_string())
    print(f"\ntest側の曖昧帯: {int(((pte > args.lo) & (pte < args.hi)).sum())} / {len(pte)}")

    print("\n" + "=" * 78)
    print("2. 誰が帯に落ちるか")
    print("=" * 78)
    for c in ["業界", "上場種別", "特徴"]:
        ct = pd.crosstab(d[c], d["_g"])
        ct["計"] = ct.sum(1)
        ct["曖昧率"] = (ct["曖昧帯"] / ct["計"]).round(3)
        ct = ct[ct["計"] >= (15 if c == "業界" else 1)]
        print(f"\n[{c}]")
        print(ct.sort_values("曖昧率", ascending=False).to_string())
    cols = ["従業員数", "事業所数", "工場数", "店舗数", "売上", "営業利益", "総資産",
            "無形固定資産変動(ソフトウェア関連)"]
    print("\n[数値プロファイル（中央値）]")
    print(d.groupby("_g")[cols].median().T.round(1).to_string())

    rng = np.random.default_rng(0)
    print("\n" + "=" * 78)
    print("3. 帯の中で購入/非購入を分ける生列（95%CI）")
    print("=" * 78)
    num = d[band].select_dtypes(include=[np.number]).drop(columns=["企業ID", "購入フラグ"])
    rows = []
    for c in num.columns:
        r = band_auc_ci(d[c].values, yb, band, rng)
        if r:
            rows.append((c, r[0], r[1], r[2], r[3], r[1] > 0.5 or r[2] < 0.5))
    df = pd.DataFrame(rows, columns=["列", "帯内AUC", "lo95", "hi95", "n", "有意95"])
    df["d"] = (df["帯内AUC"] - 0.5).abs()
    print(df.sort_values("d", ascending=False).head(10).drop(columns="d")
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78)
    print("4. 帯の中での各Expertの識別力")
    print("=" * 78)
    for n in EXPERTS:
        r = band_auc_ci(oof[n], yb, band, rng)
        print(f"  {n:14s} AUC {r[0]:.4f}  95%CI [{r[1]:.3f}, {r[2]:.3f}]"
              f"  {'' if (r[1] > 0.5 or r[2] < 0.5) else '← コイン投げと区別できない'}")

    print("\n" + "=" * 78)
    print("5. ドメイン仮説特徴の帯内スクリーニング")
    print("=" * 78)
    F = domain_features(d)
    rows = []
    for k, v in F.items():
        r = band_auc_ci(v, yb, band, rng)
        if r:
            rows.append((k, r[0], r[1], r[2], r[3],
                         r[1] > 0.5 or r[2] < 0.5, r[4] > 0.5 or r[5] < 0.5))
    df = pd.DataFrame(rows, columns=["特徴", "帯内AUC", "lo95", "hi95", "n", "有意95", "有意_多重補正"])
    df["d"] = (df["帯内AUC"] - 0.5).abs()
    print(f"検定数 {len(df)} 本。多重補正は99.5%CI（Bonferroni近似）\n")
    print(df.sort_values("d", ascending=False).drop(columns="d")
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78)
    print("6. 決定的テスト — 現本命スコアの上に足すと帯内で動くか（5seed CV）")
    print("=" * 78)
    lp = np.log(p / (1 - p))[band].reshape(-1, 1)

    def inc(extra):
        X = lp if extra is None else np.column_stack([lp, np.asarray(extra, float)[band]])
        X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
        au, apv = [], []
        for s in range(5):
            o = np.zeros(len(yb))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(X, yb):
                m = LogisticRegression(max_iter=3000, random_state=0).fit(X[a], yb[a])
                o[b] = m.predict_proba(X[b])[:, 1]
            au.append(roc_auc_score(yb, o))
            apv.append(average_precision_score(yb, o))
        return np.array(au), np.array(apv)

    b_au, b_ap = inc(None)
    print(f"  {'現本命スコアのみ':<30} 帯内AUC {b_au.mean():.4f}  帯内AP {b_ap.mean():.4f}")
    top = df.sort_values("d", ascending=False)["特徴"].head(6).tolist()
    for k in top:
        au, apv = inc(F[k])
        dau, dap = au - b_au, apv - b_ap
        print(f"  + {k:<28} 帯内AUC {au.mean():.4f} ({dau.mean():+.4f}, {int((dau > 0).sum())}/5)"
              f"  帯内AP {apv.mean():.4f} ({dap.mean():+.4f}, {int((dap > 0).sum())}/5)")
    print("\n  APが下がる候補は、単変量で有意でも採用しない（exp020/023/025と同型）。")

    print("\n" + "=" * 78)
    print("7. 42列の外＝テキストにしか無いドメイン情報は帯で効くか")
    print("=" * 78)
    dx = d["今後のDX展望"].astype(str)
    print(f"帯全体の購入率 {yb.mean():.3f}\n")
    print(f"{'テキスト内の言及':<20}{'帯内出現率':>10}{'言及ありの購入率':>12}"
          f"{'言及なしの購入率':>12}{'帯内AUC':>9}  95%CI")
    for k, pat in TEXT_MARKERS.items():
        v = dx.str.contains(pat, regex=True).astype(int).values
        vb = v[band]
        if vb.sum() < 8 or vb.sum() > len(vb) - 8:
            print(f"{k:<20}{vb.mean():>10.3f}{'':>12}{'':>12}{'':>9}  出現率が偏りすぎて判定不能")
            continue
        r = band_auc_ci(v, yb, band, rng, n_boot=3000)
        flag = "  ← 有意" if (r[1] > 0.5 or r[2] < 0.5) else ""
        print(f"{k:<20}{vb.mean():>10.3f}{yb[vb == 1].mean():>12.3f}"
              f"{yb[vb == 0].mean():>12.3f}{r[0]:>9.3f}  [{r[1]:.3f}, {r[2]:.3f}]{flag}")


if __name__ == "__main__":
    main()
