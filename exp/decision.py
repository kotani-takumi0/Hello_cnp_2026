"""判定ルール。ノイズ実測(15seed)に基づく数値基準。
乱数1本のペア差分: ΔAP 正0/15, ΔAUC 正3/15, ΔF1 正4/15(単seed最大+0.013)
→ F1単seedは判定に使わない。AP を主、AUC を確認、F1 はガードレール。
"""
import numpy as np

import math

N_SEEDS = 15  # 本判定。小規模検証時は3〜6に下げる
# 指標の役割: AUC=検出(高感度), AP=妥当性(F1と同じ正例側), F1=ガードレール
# sign閾値はseed数に追従させるため「正の割合」で定義。
FRAC_DETECT_AUC = 0.70   # ΔAUC正の割合がこれ以上で「実シグナルあり」
FRAC_RELEVANT_AP = 0.75  # ΔAP正の割合がこれ以上で「F1に効く」
MIN_MEAN_AP_CLEAR = 0.005   # 平均ΔAPがこれ以上なら「明確採用」
GUARD_F1 = -0.002           # 平均ΔF1 がこれを下回ったら不採用（F1悪化）


def _need(frac, n):
    return math.ceil(frac * n)


def paired_deltas(base_df, cand_df):
    """同一seed順で並んだ2つのスコア表のペア差分。"""
    d = {}
    for k in ["auc", "ap", "f1"]:
        diff = cand_df[k].values - base_df[k].values
        d[k] = dict(
            mean=float(diff.mean()), std=float(diff.std()),
            n_pos=int((diff > 0).sum()), n=len(diff), max=float(diff.max()),
            min=float(diff.min()),
        )
    return d


def verdict(base_df, cand_df):
    d = paired_deltas(base_df, cand_df)
    ap, auc, f1 = d["ap"], d["auc"], d["f1"]

    n = auc["n"]
    guard_ok = f1["mean"] >= GUARD_F1
    detect = auc["n_pos"] >= _need(FRAC_DETECT_AUC, n)          # 実シグナルの有無
    relevant = ap["n_pos"] >= _need(FRAC_RELEVANT_AP, n) and ap["mean"] > 0  # F1に効くか

    if not guard_ok:
        v = "REJECT (F1悪化: ガードレール違反)"
    elif detect and relevant and ap["mean"] >= MIN_MEAN_AP_CLEAR:
        v = "ACCEPT (明確)"
    elif detect and relevant:
        v = "ACCEPT (弱い)"
    elif detect and not relevant:
        v = "PARK (AUCは改善だがF1関連のAPが未達→保留・組合せで再検証)"
    else:
        v = "REJECT (ノイズと区別できず)"

    return v, d


def format_report(name, desc, base_df, cand_df, v, d):
    L = [f"仮説 {name}: {desc}",
         f"  特徴数: {base_df.attrs.get('n_feat','?')} -> {cand_df.attrs.get('n_feat','?')}",
         f"  {'指標':4s}  baseline        候補            Δmean    Δstd    正/n   単seed最大Δ"]
    for k, lab in [("ap", "AP"), ("auc", "AUC"), ("f1", "F1")]:
        b, c, x = base_df[k], cand_df[k], d[k]
        L.append(f"  {lab:4s}  {b.mean():.4f}±{b.std():.4f}  "
                 f"{c.mean():.4f}±{c.std():.4f}  {x['mean']:+.4f}  "
                 f"{x['std']:.4f}  {x['n_pos']:2d}/{x['n']}   {x['max']:+.4f}")
    L.append(f"  判定 => {v}")
    return "\n".join(L)
