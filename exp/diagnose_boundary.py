"""H25: 閾値**付近の行**に手が入る余地があるかを測る（着手前の上限確認）。

exp024 は「閾値の値の決め方」を4則で潰した（oof_best 据え置き）。ここで見るのは別物で、
**決めた閾値のすぐ両側に居る行そのもの**に構造が残っているかどうか。

順序:
  1. 現本命(exp026)の合成OOFで t* とプラトーを出し、境界帯を定義する
  2. **上限を先に測る**。帯の中を全部正解にできたら F1 はいくつになるか。
     ここが現行と大差なければ、この路線は何をやっても届かないので即終了する
  3. 上限が十分あるときだけ、帯の中に構造があるかを探す
     （エキスパート間の不一致 / 各エキスパートのlogit / カテゴリ）

**方法論上の注意**: 帯は予測が最も割れている領域＝最高分散の場所で、そこの
OOFラベルを見て何かを選ぶのは定義上いちばん過学習しやすい。だから探索段階では
必ずラベル置換のヌルと並べて出す。n≈100 の帯では「それらしい差」は何もなくても出る。

  python exp/diagnose_boundary.py [--seed 42] [--band-rows 40]

出力: exp/_h25_boundary.csv （帯の中の行ごとの内訳）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import EXPERTS, blend_oof, compute_expert_preds  # noqa: E402
from meta_blend import to_logit  # noqa: E402
from threshold import THS, analyze, f1_at, f1_curve  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def band_by_rank(p, th, k):
    """閾値の順位上の前後 k 行を境界帯とする。

    確率の絶対値でなく順位で取るのは、帯の大きさを構成によらず一定にするため。
    """
    order = np.argsort(-p)
    cut = int((p >= th).sum())
    lo, hi = max(0, cut - k), min(len(p), cut + k)
    idx = np.zeros(len(p), dtype=bool)
    idx[order[lo:hi]] = True
    return idx, cut


def ceiling(y, p, th, band):
    """帯の中を全部正解にできたときの F1（この路線の理論上限）。"""
    pred = (p >= th).astype(int)
    fixed = np.where(band, y, pred)
    return _f1(y, pred), _f1(y, fixed)


def _f1(y, pred):
    tp = int((pred & y).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (int(pred.sum()) + int(y.sum()))


def null_auc(score, lab, n_perm=2000, seed=0):
    """ラベル置換で作る「帯の中のAUC」のヌル分布。実測値の位置を返す。"""
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(lab, score)
    null = np.array([roc_auc_score(rng.permutation(lab), score)
                     for _ in range(n_perm)])
    two_sided = float((np.abs(null - 0.5) >= abs(obs - 0.5)).mean())
    return obs, float(null.std()), two_sided


def sweep(seed, org_embed, band_list, n_perm):
    """帯幅を振って「上限」と「帯の中の信号」だけを要約する（構成×seedの横断用）。

    exp024 の教訓（閾値まわりの判定は片方の構成だけ見ると逆の結論に飛べる）に従い、
    exp026構成と exp019構成の両方で同じ表を出せるようにしてある。
    """
    y, oof, te = compute_expert_preds(seed, org_embed=org_embed)
    blend, _, _, _ = blend_oof(y, oof, te, seed, alpha=None)
    th = analyze(y, blend[None, :])["th_star"]
    Z = to_logit(np.column_stack([oof[n] for n in EXPERTS]))

    rows = []
    for k in band_list:
        band, _ = band_by_rank(blend, th, k)
        lab = y[band].astype(int)
        now, top = ceiling(y, blend, th, band)
        auc_blend, _, p_blend = null_auc(blend[band], lab, n_perm)
        dis = Z[band].max(1) - Z[band].min(1)
        auc_dis, _, p_dis = null_auc(dis, lab, n_perm)
        rows.append(dict(
            seed=seed, 構成="exp026" if org_embed else "exp019", 帯幅=2 * k,
            n=int(band.sum()), 帯の正例率=round(lab.mean(), 3),
            現行F1=round(now, 4), 上限F1=round(top, 4), 余地=round(top - now, 4),
            帯内AUC_合成=round(auc_blend, 3), p_合成=round(p_blend, 3),
            帯内AUC_不一致=round(auc_dis, 3), p_不一致=round(p_dis, 3)))
    return pd.DataFrame(rows)


def run(seed, band_rows, n_perm, org_embed=True):
    y, oof, te = compute_expert_preds(seed, org_embed=org_embed)
    blend, _, w, _ = blend_oof(y, oof, te, seed, alpha=None)

    a = analyze(y, blend[None, :])
    th = a["th_star"]
    tag = "exp026" if org_embed else "exp019"
    print(f"=== {tag} 合成OOF / seed={seed} ===")
    print(f"  t* = {th:.3f}  OOF F1 = {a['f1_insample']:.4f}  "
          f"プラトー [{a['plateau'][0]:.3f}, {a['plateau'][1]:.3f}]")

    band, cut = band_by_rank(blend, th, band_rows)
    n_band = int(band.sum())
    pos_in = int(y[band].sum())
    print(f"\n=== 境界帯: 閾値の順位前後 {band_rows} 行 ===")
    print(f"  帯 {n_band}行 / train {len(y)}行 ({n_band / len(y):.1%})"
          f"  予測正例カット位置 {cut}")
    print(f"  帯の中の実正例 {pos_in}/{n_band} = {pos_in / n_band:.3f}"
          f"  (全体の正例率 {y.mean():.3f})")

    now, top = ceiling(y, blend, th, band)
    print(f"\n=== 上限（帯を全部正解にできた場合）===")
    print(f"  現行 F1 {now:.4f}  ->  上限 F1 {top:.4f}   余地 {top - now:+.4f}")
    print(f"  参考: Public F1 の実測std は 0.043。上限がこれに届かないなら"
          f"\n        この路線は測定ノイズに埋もれる")

    # 帯の中だけを見たときに、まだ順位付けの余地が残っているか
    lab = y[band].astype(int)
    print(f"\n=== 帯の中で合成スコア自身はまだ効いているか ===")
    obs, sd, p_val = null_auc(blend[band], lab, n_perm)
    print(f"  合成スコア      AUC {obs:.4f}  (ヌルsd {sd:.4f}, 置換p {p_val:.3f})")
    print("  → 0.5 近傍なら『帯の中は合成モデルにとって完全に無情報』")

    print(f"\n=== 帯の中で他の信号は効くか（ラベル置換 {n_perm}回のヌルと比較）===")
    Z = to_logit(np.column_stack([oof[n] for n in EXPERTS]))
    cand = {n: Z[band, i] for i, n in enumerate(EXPERTS)}
    cand["不一致(std)"] = Z[band].std(1)
    cand["不一致(range)"] = Z[band].max(1) - Z[band].min(1)
    rows = []
    for name, s in cand.items():
        obs, sd, p_val = null_auc(s, lab, n_perm)
        flag = "  <-- 有意" if p_val < 0.05 else ""
        print(f"  {name:14s} AUC {obs:.4f}  ヌルsd {sd:.4f}  置換p {p_val:.3f}{flag}")
        rows.append(dict(signal=name, auc=obs, null_sd=sd, perm_p=p_val))

    n_sig = sum(r["perm_p"] < 0.05 for r in rows)
    print(f"\n  {len(rows)}本中 {n_sig}本が p<0.05（多重比較の期待値 "
          f"{0.05 * len(rows):.1f}本）")

    out = pd.DataFrame(dict(
        idx=np.where(band)[0], y=lab, blend=blend[band],
        **{n: Z[band, i] for i, n in enumerate(EXPERTS)},
        disagree_std=Z[band].std(1)))
    out.to_csv(os.path.join(OUT_DIR, "_h25_boundary.csv"), index=False)
    print(f"\n保存: {OUT_DIR}/_h25_boundary.csv")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "_h25_boundary_signals.csv"),
                              index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band-rows", type=int, default=40,
                   help="閾値の順位上、前後何行を境界帯とみなすか")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--no-org-embed", action="store_true",
                   help="exp019構成(E4がTF-IDF)で見る。キャッシュ済みseedが多い")
    p.add_argument("--sweep", nargs="*", type=int,
                   help="帯幅を振って要約表だけ出す（引数は片側行数）")
    p.add_argument("--sweep-seeds", nargs="*", type=int, default=None,
                   help="--sweep で横断するseed。既定は --seed のみ")
    a = p.parse_args()
    org = not a.no_org_embed
    if a.sweep is not None:
        band_list = a.sweep or [20, 40, 60, 80]
        seeds = a.sweep_seeds or [a.seed]
        df = pd.concat([sweep(s, org, band_list, a.n_perm) for s in seeds],
                       ignore_index=True)
        print(df.to_string(index=False))
        out = os.path.join(OUT_DIR, "_h25_boundary_sweep.csv")
        df.to_csv(out, index=False)
        print(f"\n保存: {out}")
    else:
        run(a.seed, a.band_rows, a.n_perm, org_embed=org)
