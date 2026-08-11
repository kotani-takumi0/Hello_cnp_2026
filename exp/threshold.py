"""閾値の決定と、その不確実性の測定。

F1 は AUC/AP と違って閾値というパラメータを含む。OOF で argmax した時点で
それは「OOFに合わせ込んだパラメータ」であり、必ず楽観バイアスを持つ
（exp004 が Public で崩れた原因）。ここではそのバイアスを測ってから閾値を決める。

方針:
  1. seed ごとの argmax は使わない。**F1(t) 曲線を平均してから argmax** する。
     平坦な曲面のノイズの頂点ではなく、プラトーの中心を取るため。
  2. `leave-one-seed-out` で「知らない閾値」での F1 を測り、選択の下駄を定量化する。
     報告すべき CV スコアは in-sample の max ではなくこちら。
  3. test へは絶対閾値ではなく **正例率 r\\*** を転写する（レートマッチング）。
     OOF の各行は 1 モデルの出力、test の各行は 5×n_seeds モデルの平均で、
     平均は分散を縮めるので確率スケールが揃わない。r\\* は順位ベースなので
     この収縮に不変で、モデル構成を変えても t\\* よりはるかに動かない。
"""
import numpy as np

THS = np.arange(0.02, 0.98, 0.002)
PLATEAU_TOL = 0.005


def f1_curve(y, p, ths=THS):
    """全閾値の F1 を一括計算（ベクトル化, 厳密値）。"""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp_cum = np.cumsum(ys)
    n_true = ys.sum()
    if n_true == 0:
        return np.zeros(len(ths))
    n_pos = np.searchsorted(-ps, -ths, side="right")      # p >= t の件数
    tp = np.where(n_pos > 0, tp_cum[np.clip(n_pos - 1, 0, None)], 0)
    prec = np.where(n_pos > 0, tp / np.maximum(n_pos, 1), 0.0)
    rec = tp / n_true
    denom = prec + rec
    return np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)


def f1_at(y, p, t):
    return float(f1_curve(y, p, np.array([t]))[0])


def curves(y, OOF, ths=THS):
    """(n_seeds, n_ths) の F1 曲線行列。"""
    return np.vstack([f1_curve(y, OOF[s], ths) for s in range(OOF.shape[0])])


def plateau(curve, ths=THS, tol=PLATEAU_TOL):
    """最大から tol 以内に収まる閾値の範囲。"""
    inside = np.where(curve >= curve.max() - tol)[0]
    return float(ths[inside.min()]), float(ths[inside.max()])


def analyze(y, OOF, ths=THS, tol=PLATEAU_TOL):
    """閾値の解析一式。採用する th\\* と r\\*、および正直な F1 推定を返す。"""
    C = curves(y, OOF, ths)
    n_seeds = C.shape[0]
    mean_curve = C.mean(0)
    i_star = int(np.argmax(mean_curve))
    th_star = float(ths[i_star])
    f1_insample = float(mean_curve[i_star])

    # leave-one-seed-out: 他 seed で決めた閾値を当該 seed に当てる
    loso_th, loso_f1 = [], []
    for s in range(n_seeds):
        others = np.delete(np.arange(n_seeds), s)
        t_hat = float(ths[int(np.argmax(C[others].mean(0)))]) if n_seeds > 1 else th_star
        loso_th.append(t_hat)
        loso_f1.append(f1_at(y, OOF[s], t_hat))
    loso_f1 = np.array(loso_f1)

    per_seed_th = ths[C.argmax(1)]
    rates = np.array([(OOF[s] >= th_star).mean() for s in range(n_seeds)])
    lo, hi = plateau(mean_curve, ths, tol)

    return dict(
        ths=ths, curves=C, mean_curve=mean_curve, sd_curve=C.std(0),
        th_star=th_star, f1_insample=f1_insample,
        f1_loso=float(loso_f1.mean()), f1_loso_std=float(loso_f1.std()),
        selection_bias=float(f1_insample - loso_f1.mean()),
        loso_th=np.array(loso_th),
        per_seed_th=per_seed_th, per_seed_f1=C.max(1),
        plateau=(lo, hi), plateau_width=hi - lo,
        rate=float(rates.mean()), rate_std=float(rates.std()),
        base_rate=float(y.mean()),
        calib_ref=f1_insample / 2.0,   # 校正が正しければ th* ≒ F1/2
    )


def fold_threshold_std(y, OOF, FOLD, ths=THS):
    """fold ごとに独立に閾値を決めた場合のばらつき（seed 平均）。"""
    out = []
    for s in range(OOF.shape[0]):
        per_fold = [float(ths[int(np.argmax(f1_curve(y[FOLD[s] == k],
                                                     OOF[s][FOLD[s] == k], ths)))])
                    for k in np.unique(FOLD[s])]
        out.append(np.std(per_fold))
    return float(np.mean(out))


def bootstrap_threshold(y, OOF, n_boot=1000, ths=THS, seed=0, max_seeds=5):
    """行のリサンプリングだけで閾値がどれだけ動くか（データ量由来のノイズ）。"""
    rng = np.random.default_rng(seed)
    use = min(max_seeds, OOF.shape[0])
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = np.vstack([f1_curve(y[idx], OOF[s][idx], ths) for s in range(use)]).mean(0)
        out[b] = ths[int(np.argmax(m))]
    return out


def rate_match_cut(test_proba, rate):
    """test の上位 rate 割を正例にする切り口（分位点）。"""
    return float(np.quantile(test_proba, 1.0 - rate))


def format_report(a, label="", fold_std=None, boot=None):
    L = [f"=== 閾値解析 {label} ===",
         f"  採用 th*            : {a['th_star']:.3f}",
         f"  in-sample 平均OOF F1: {a['f1_insample']:.4f}  (楽観・報告に使わない)",
         f"  LOSO 正直な F1      : {a['f1_loso']:.4f} ± {a['f1_loso_std']:.4f}  <= これを報告",
         f"  選択バイアス        : {a['selection_bias']:+.4f}",
         f"  seed毎 argmax       : mean {a['per_seed_th'].mean():.3f} "
         f"std {a['per_seed_th'].std():.3f} "
         f"range [{a['per_seed_th'].min():.3f}, {a['per_seed_th'].max():.3f}]",
         f"  プラトー(-{PLATEAU_TOL})    : [{a['plateau'][0]:.3f}, {a['plateau'][1]:.3f}] "
         f"幅 {a['plateau_width']:.3f}",
         f"  th* での正例率 r*   : {a['rate']:.4f} ± {a['rate_std']:.4f} "
         f"(学習 {a['base_rate']:.4f})",
         f"  校正チェック        : 理論 th≒F1/2={a['calib_ref']:.3f} vs 実測 "
         f"{a['th_star']:.3f} "
         f"({'確率が縮んでいる→レートマッチ推奨' if a['th_star'] < a['calib_ref'] - 0.02 else 'おおむね整合'})"]
    if fold_std is not None:
        L.append(f"  fold内閾値std       : {fold_std:.4f} (既知: baseline 0.073 / H8 0.111)")
    if boot is not None:
        L.append(f"  bootstrap th        : 中央 {np.median(boot):.3f} 90%CI "
                 f"[{np.percentile(boot, 5):.3f}, {np.percentile(boot, 95):.3f}]")
    return "\n".join(L)
