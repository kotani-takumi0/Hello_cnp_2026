"""ベースモデルの統一インターフェース（LGBM / CatBoost）。

すべてのモデルは同じシグネチャを持つ:

    fit_predict(Xtr, ytr, Xva, yva, Xte, seed) -> dict(va=, te=, best_iter=, gain=)

同一 fold・同一特徴量を食わせて (valid予測, test予測) だけを返させることで、
呼び出し側は「行単位で揃った複数モデルの予測」を得る。これが揃っていないと
ペア比較もブレンドもできない。

容量を揃えてある点が重要:
  LGBM     num_leaves=15        （深さ4相当の非対称木）
  CatBoost depth=4 → 16葉       （対称木）
これで「アルゴリズムの違い」だけが差として出る。片方だけ表現力を上げると
アンサンブルの利得なのか単にモデルが強いだけなのか切り分けられなくなる。

CatBoost は Colab 側でのみ使う想定なので import は遅延。ローカルに catboost が
無くても LGBM だけのスモークは回る。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

CAT_COLS = ["業界", "上場種別", "特徴"]
NUM_ROUNDS = 2000
EARLY_STOPPING = 100
CAT_NA = "__NA__"

# exp001〜007 と同一（seed だけ呼び出し時に差し込む）
LGBM_PARAMS = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.03,
    num_leaves=15, min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
)

# LGBM と容量・学習率・サンプリング率を揃えた設定。
# CatBoost 固有の差分は (a) 対称木 (b) カテゴリの ordered target statistics
# (c) ordered boosting。742行の小データでは (b)(c) の正則化が効きやすい。
CATBOOST_PARAMS = dict(
    loss_function="Logloss", eval_metric="Logloss",
    learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
    bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
    iterations=NUM_ROUNDS, allow_writing_files=False, verbose=False,
)


def _seeded(params, seed, key):
    return {**params, key: seed}


def fit_predict_lgbm(Xtr, ytr, Xva, yva, Xte, seed):
    params = dict(LGBM_PARAMS, seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed, data_random_seed=seed)
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=NUM_ROUNDS,
                  valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    gain = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    return dict(va=m.predict(Xva, num_iteration=m.best_iteration),
                te=m.predict(Xte, num_iteration=m.best_iteration),
                best_iter=int(m.best_iteration), gain=gain)


def _catboost_frame(X):
    """カテゴリ列を欠損なしの文字列に変換した新しい DataFrame を返す。

    CatBoost の cat_features は NaN を受け付けない。pandas.Categorical に無い値は
    NaN になっているので、明示的に専用カテゴリへ寄せる（test 側の未知カテゴリも
    ここで CAT_NA に落ちる = LGBM が NaN 扱いするのと同じ意味になる）。
    数値列の NaN は CatBoost がネイティブに扱うのでそのまま。
    """
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            s = X[c].astype(object)
            X[c] = s.where(s.notna(), CAT_NA).astype(str)
    return X


def fit_predict_catboost(Xtr, ytr, Xva, yva, Xte, seed):
    from catboost import CatBoostClassifier  # Colab 側でのみ必要

    Xtr, Xva, Xte = _catboost_frame(Xtr), _catboost_frame(Xva), _catboost_frame(Xte)
    cats = [c for c in CAT_COLS if c in Xtr.columns]
    m = CatBoostClassifier(**_seeded(CATBOOST_PARAMS, seed, "random_seed"),
                           cat_features=cats)
    m.fit(Xtr, ytr, eval_set=(Xva, yva),
          early_stopping_rounds=EARLY_STOPPING, use_best_model=True, verbose=False)
    gain = pd.Series(m.get_feature_importance(), index=list(Xtr.columns))
    return dict(va=m.predict_proba(Xva)[:, 1],
                te=m.predict_proba(Xte)[:, 1],
                best_iter=int(m.get_best_iteration()), gain=gain)


MODEL_REGISTRY = {
    "lgbm": ("LightGBM (num_leaves=15)", fit_predict_lgbm),
    "catboost": ("CatBoost (depth=4, ordered TS)", fit_predict_catboost),
}


def available(names):
    """未インストールのモデルを弾いた名前リストを返す。"""
    ok = []
    for n in names:
        if n not in MODEL_REGISTRY:
            raise KeyError(f"未知のモデル: {n} (候補: {list(MODEL_REGISTRY)})")
        if n == "catboost":
            try:
                import catboost  # noqa: F401
            except ImportError:
                print(f"[skip] catboost 未インストールのため {n} を除外")
                continue
        ok.append(n)
    if not ok:
        raise RuntimeError("実行可能なモデルが1つも無い")
    return ok
