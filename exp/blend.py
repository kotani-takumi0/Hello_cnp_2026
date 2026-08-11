"""複数モデルの予測を混ぜる。重みは **フィットしない**。

742行しかないので、OOF 上で重みを最適化するのは閾値と全く同じ過学習の穴を
もう一つ開ける行為になる。既定は等重み。重み掃引は「最適が本当に 0.5 付近か」を
目で確認するための診断であって、そこから重みを採用するためのものではない。

混ぜ方は rank 平均を既定にする。LGBM と CatBoost は確率のスケールが揃わない
（CatBoost の ordered boosting は確率を保守的に出しがち）ので、確率平均だと
自信過剰なモデルに引きずられる。rank ならスケール差に不変で、レートマッチングと
同じ「順位しか見ない」土俵に乗る。
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

from threshold import THS, f1_curve, fold_threshold_std


def rank_norm(mat):
    """(n_seeds, n) の各行を [0,1) の順位に正規化する。"""
    mat = np.atleast_2d(mat)
    return rankdata(mat, axis=1) / (mat.shape[1] + 1.0)


def blend(mats, weights=None, method="rank"):
    """行列のリストを混ぜて (n_seeds, n) を返す。

    mats   : [(n_seeds, n), ...] 各モデルの確率行列（OOF か TEST のどちらか片方）
    method : "rank" 順位平均 / "prob" 確率平均
    OOF と TEST は必ず別々に呼ぶこと。母集団が違うので同じ順位空間に載せてはいけない。
    """
    if weights is None:
        weights = np.ones(len(mats))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    parts = [rank_norm(m) if method == "rank" else np.atleast_2d(m) for m in mats]
    return sum(wi * p for wi, p in zip(w, parts))


def seed_metrics(y, OOF, FOLD=None, ths=THS, n_feat=None):
    """seed ごとの指標表。decision.verdict にそのまま渡せる形。"""
    rows = []
    for s in range(OOF.shape[0]):
        p = OOF[s]
        curve = f1_curve(y, p, ths)
        i = int(np.argmax(curve))
        rows.append(dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                         f1=float(curve[i]), th=float(ths[i])))
    df = pd.DataFrame(rows)
    if FOLD is not None:
        df["th_fold_std"] = fold_threshold_std(y, OOF, FOLD, ths)
    if n_feat is not None:
        df.attrs["n_feat"] = n_feat
    return df


def rank_correlation(OOF_a, OOF_b):
    """seed ごとの Spearman 相関の平均。ブレンド前のスクリーニングに使う。

    0.95 を超えるなら 2 モデルは実質同じ順位を出しており、混ぜても何も起きない。
    その場合はアンサンブルではなく多様性の源（テキスト表現など）を変えるべき。
    """
    ra, rb = rank_norm(OOF_a), rank_norm(OOF_b)
    return float(np.mean([np.corrcoef(ra[s], rb[s])[0, 1] for s in range(ra.shape[0])]))


def disagreement(OOF_a, OOF_b, y, ths=THS):
    """各 seed で両モデルを各自の最適閾値で2値化したときのラベル不一致率。"""
    out = []
    for s in range(OOF_a.shape[0]):
        la = _labels_at_best(y, OOF_a[s], ths)
        lb = _labels_at_best(y, OOF_b[s], ths)
        out.append((la != lb).mean())
    return float(np.mean(out))


def _labels_at_best(y, p, ths):
    t = ths[int(np.argmax(f1_curve(y, p, ths)))]
    return (p >= t).astype(int)


def weight_sweep(y, OOF_a, OOF_b, method="rank", step=0.1, ths=THS):
    """w を振って AP と「正直な F1」を見る診断表。採用のためではなく確認のため。

    F1 は leave-one-seed-out で閾値を決めて評価する。in-sample の max を並べると
    どの w でも一律に下駄が乗り、形が読めなくなるため。
    """
    rows = []
    for w in np.round(np.arange(0.0, 1.0 + 1e-9, step), 3):
        B = blend([OOF_a, OOF_b], weights=[w, 1 - w], method=method)
        C = np.vstack([f1_curve(y, B[s], ths) for s in range(B.shape[0])])
        loso = []
        for s in range(B.shape[0]):
            others = np.delete(np.arange(B.shape[0]), s)
            t = ths[int(np.argmax(C[others].mean(0)))] if B.shape[0] > 1 else \
                ths[int(np.argmax(C[s]))]
            loso.append(f1_curve(y, B[s], np.array([t]))[0])
        rows.append(dict(w_a=w,
                         ap=float(np.mean([average_precision_score(y, B[s])
                                           for s in range(B.shape[0])])),
                         auc=float(np.mean([roc_auc_score(y, B[s])
                                            for s in range(B.shape[0])])),
                         f1_loso=float(np.mean(loso))))
    return pd.DataFrame(rows)


def format_sweep(df, name_a, name_b):
    L = [f"  w({name_a})  AP       AUC      F1(LOSO)"]
    best_ap = df["ap"].idxmax()
    for i, r in df.iterrows():
        mark = "  <= AP最大" if i == best_ap else ""
        L.append(f"    {r['w_a']:.2f}    {r['ap']:.4f}   {r['auc']:.4f}   "
                 f"{r['f1_loso']:.4f}{mark}")
    L.append(f"  ({name_b} の重みは 1-w)")
    return "\n".join(L)
