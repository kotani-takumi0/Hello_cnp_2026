"""エキスパート予測を合成するメタ学習器。

742行に対して構成員は6本程度なので、自由度を絞った合成しか許されない。
山登りやOptunaでOOFを直接最適化するとOOFに貼り付くため使わない。

  logit(p_k) を非負重み w_k で線形結合し、sigmoid で確率に戻す。
  loss = 平均ロジスティック損失 + alpha * ペナルティ,  s.t. w_k >= 0

非負制約の意図: 742行だと「あるエキスパートを逆張りで使う」符号反転が平気で
起きる。制約を入れると、効かない構成員は重み0に落ちるだけで済む。

縮小先 (`target`):
  "zero"    ||w||^2       … 現行。**等分には近づかない**（既定、過去実験の再現用）
  "uniform" ||w - w̄・1||^2 … 重みの「ばらつき」だけを罰する

**重要（2026-08-10 の実測）**: "zero" は大きさ(||w||)を縮めるだけで、配分は
等分に向かわない。alpha→∞ での極限方向は各エキスパートの y との単独共分散に
比例するため、単体最強の E0 が支配し、**単体最弱だが直交している E2 が
真っ先に潰れる**（E2 37%→5% / ばらつき 11.0→16.9 %pt）。E2 の価値は同時推定で
しか現れない直交性なので、これは合成の効きを壊す方向。`select_alpha` が常に
グリッド下端(0.001〜0.01)を選んでいたのはこのため＝実質的に正則化は効いていない。

"uniform" なら alpha→∞ で「温度付きの等分ブレンド」に収束するので、
メタ(ホールドアウトF1 0.7555)と等分(0.7497)という**両端とも実績のある2点**を
連続的に結べる。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
# alpha=0 は入れない。OOF掃引では alpha->0 で単調に良く見えるが、それは
# 「評価データを見て選んでいる」ことの現れなので、選択自体を学習側に閉じ込める。
ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
# target="uniform" 用。等分側の端まで届かせる必要があるので上を伸ばす。
UNIFORM_ALPHA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def to_logit(p):
    """確率をlogitへ。0/1に貼り付いた値でinfにならないようclipする。"""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _penalty(w, target):
    """(値, 勾配) を返す。

    zero    : ||w||^2            grad 2w
    uniform : ||w - w̄・1||^2      grad 2(w - w̄)   （中心化行列Mが冪等なので）
    """
    if target == "zero":
        return w @ w, 2.0 * w
    if target == "uniform":
        d = w - w.mean()
        return d @ d, 2.0 * d
    raise ValueError(f"未知の縮小先: {target}")


def _loss_grad(theta, Z, y, alpha, target):
    w, b = theta[:-1], theta[-1]
    z = Z @ w + b
    p = expit(z)
    p = np.clip(p, EPS, 1.0 - EPS)
    pen, dpen = _penalty(w, target)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + alpha * pen
    r = (p - y) / len(y)
    return loss, np.concatenate([Z.T @ r + alpha * dpen, [r.sum()]])


def fit_meta(Z, y, alpha=1.0, target="zero"):
    """非負重み + 切片を返す。Z は logit 済みの (n, k) 行列。"""
    k = Z.shape[1]
    theta0 = np.concatenate([np.full(k, 1.0 / k), [0.0]])
    bounds = [(0.0, None)] * k + [(None, None)]
    res = minimize(_loss_grad, theta0, args=(Z, y, alpha, target), jac=True,
                   method="L-BFGS-B", bounds=bounds)
    return res.x[:-1], res.x[-1]


def apply_meta(Z, w, b):
    return expit(Z @ w + b)


def uniform_blend(Z):
    """等分ブレンド（logit平均）。学習済み重みの比較対象。"""
    return expit(Z.mean(axis=1))


def select_alpha(Z, y, grid=None, n_splits=5, seed=0, target="zero"):
    """渡された (Z, y) の中だけで alpha を選ぶ。評価データは一切見ない。

    選択基準は AP。decision.py の方針（AUC=検出 / AP=F1妥当性 / F1=ガードレール）に
    従い、F1は単発だと揺れが大きすぎるので選択には使わない。
    """
    if grid is None:
        grid = ALPHA_GRID if target == "zero" else UNIFORM_ALPHA_GRID
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(Z, y))
    best, best_ap = grid[0], -np.inf
    for a in grid:
        oof = np.zeros(len(y))
        for tr, va in folds:
            w, b = fit_meta(Z[tr], y[tr], alpha=a, target=target)
            oof[va] = apply_meta(Z[va], w, b)
        ap = average_precision_score(y, oof)
        if ap > best_ap:
            best, best_ap = a, ap
    return best, best_ap
