"""再利用可能な評価ハーネス。
1仮説ずつ検証するための共通土台。baseline.ipynb のモデルを忠実に再現し、
複数seedで回して OOF指標の mean±std（＝ノイズ幅）を返す。

スレッド数について（2026-08-14、exp030 で踏んだ）
--------------------------------------------------
n=742・p=94 は **スレッドを増やすほど遅くなる**領域にある。742行を16スレッドに割ると
1スレッド46行で、ノードごとの待ち合わせコストのほうが計算より大きい。しかも
LightGBM の OpenMP プールと numpy/BLAS のプールが**それぞれコア数ぶん**確保するので
1プロセス47スレッドになり、2ジョブ並べると 94スレッド/16コア でスラッシングする
（実測: CPU時間79分に対し wall 9.5分で rep1/5 すら未完 ＝ CPUの8割がスピンウェイト）。

`num_threads=4` にすると1回のエキスパート学習が5.5秒になり、15repホールドアウトが
7分で終わる（**CPUは7分の1しか使わずに17倍速い**）。
結果は変わらないことを `repro_check.py` で確認済み（最大 4.9e-15 の丸め差、提出ラベル差0行）。
コア数の多いマシンで上書きしたいときは環境変数 `SIGNATE_LGB_THREADS` で指定する。
"""
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, log_loss
import warnings
warnings.filterwarnings("ignore")

TARGET = "購入フラグ"
DROP_TEXT = ["企業ID", "企業名", "企業概要", "組織図", "今後のDX展望"]
CAT_COLS = ["業界", "上場種別", "特徴"]
ZERO_FILL = ["事業所数", "工場数", "店舗数", "アンケート７"]
FLAG_ONLY = ["資本金", "営業利益", "経常利益"]

# 学習には影響しない実行時パラメータ（上の docstring 参照）。
LGB_THREADS = int(os.environ.get("SIGNATE_LGB_THREADS", "4"))

BASE_PARAMS = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.03,
    num_leaves=15, min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
    num_threads=LGB_THREADS,
)


def preprocess(df):
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_TEXT if c in df.columns])
    for c in ZERO_FILL:
        df[f"{c}_isna"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0)
    for c in FLAG_ONLY:
        df[f"{c}_isna"] = df[c].isna().astype(int)
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df


def run_cv(X, y, seed, fold_transform=None):
    """1 seed 分の OOF を作り指標を返す。
    fold_transform(X_tr, y_tr, X_va) -> (X_tr2, X_va2): fold内fitが要る特徴
    （target encoding等）用。train統計のみでfitしvalidに適用（リーク無し）。
    """
    params = {**BASE_PARAMS, "seed": seed}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    fold_th = []
    ths = np.arange(0.05, 0.95, 0.005)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if fold_transform is not None:
            Xtr, Xva = fold_transform(Xtr, y[tr], Xva)
        dtr = lgb.Dataset(Xtr, y[tr])
        dva = lgb.Dataset(Xva, y[va])
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        f = [f1_score(y[va], (oof[va] >= t).astype(int)) for t in ths]
        fold_th.append(ths[int(np.argmax(f))])
    # F1: OOF全体で最適化した閾値（LB提出時と同じ手順）
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    best_i = int(np.argmax(f1s))
    return dict(
        auc=roc_auc_score(y, oof),
        ap=average_precision_score(y, oof),
        f1=f1s[best_i],
        th=ths[best_i],
        th_fold_std=float(np.std(fold_th)),
        logloss=log_loss(y, oof),
    )


def _run_cv_star(args):
    X, y, s, ft = args
    return run_cv(X, y, s, fold_transform=ft)


def evaluate(add_features=None, seeds=range(10), label="model", verbose=True,
             fold_transform=None):
    """add_features(df)->df で特徴量を足したモデルを複数seedで評価。
    fold_transform は fold内fitが要る特徴（target encoding等）用。"""
    train = pd.read_csv("data/train.csv")
    tp = preprocess(train)
    y = tp[TARGET].values
    X = tp.drop(columns=[TARGET])
    if add_features is not None:
        X = add_features(train, X)
    seeds = list(seeds)
    res = [run_cv(X, y, s, fold_transform=fold_transform) for s in seeds]
    df = pd.DataFrame(res)
    df.attrs["n_feat"] = int(X.shape[1])
    if verbose:
        print(f"[{label}] n_seeds={len(res)}, n_feat={X.shape[1]}")
        for k in ["auc", "ap", "f1"]:
            print(f"  {k.upper():4s}: {df[k].mean():.4f} ± {df[k].std():.4f}"
                  f"  (min {df[k].min():.4f} / max {df[k].max():.4f})")
        print(f"  閾値: {df['th'].mean():.3f} ± {df['th'].std():.3f}"
              f" / fold内閾値std平均 {df['th_fold_std'].mean():.3f}")
    return df


if __name__ == "__main__":
    import sys
    # baseline のノイズ幅
    base = evaluate(label="baseline", seeds=range(15))
    base.to_csv("exp/_baseline_scores.csv", index=False)

    # ヌル比較: 乱数特徴を1本足しても改善して見えることがある
    def add_noise(train, X):
        rng = np.random.RandomState(0)
        X = X.copy()
        X["_noise"] = rng.randn(len(X))
        return X
    noise = evaluate(add_noise, label="baseline+乱数1本", seeds=range(15))

    print("\n=== ペア差分（同seedで baseline vs +乱数）===")
    for k in ["auc", "ap", "f1"]:
        d = noise[k].values - base[k].values
        print(f"  Δ{k.upper():4s}: mean {d.mean():+.4f}, std {d.std():.4f}, "
              f"正の回数 {int((d>0).sum())}/{len(d)}, max {d.max():+.4f}")
