"""exp002: baseline + H1(利益率3本) で提出ファイルを作る。
exp001と同一手順（SEED=42, 5fold, OOF閾値, fold平均）にH1特徴のみ追加。
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
TARGET = "購入フラグ"
DROP_TEXT = ["企業ID", "企業名", "企業概要", "組織図", "今後のDX展望"]
CAT_COLS = ["業界", "上場種別", "特徴"]
ZERO_FILL = ["事業所数", "工場数", "店舗数", "アンケート７"]
FLAG_ONLY = ["資本金", "営業利益", "経常利益"]
PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
              num_leaves=15, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, seed=SEED)


def _safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], np.nan)


def add_h1(df, X):
    """H1: 規模で正規化した利益率3本。"""
    X = X.copy()
    X["営業利益率"] = _safe_div(df["営業利益"], df["売上"])
    X["経常利益率"] = _safe_div(df["経常利益"], df["売上"])
    X["純利益率"] = _safe_div(df["当期純利益"], df["売上"])
    return X


def preprocess(df, cat_categories=None):
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_TEXT if c in df.columns])
    for c in ZERO_FILL:
        df[f"{c}_isna"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0)
    for c in FLAG_ONLY:
        df[f"{c}_isna"] = df[c].isna().astype(int)
    for c in CAT_COLS:
        if cat_categories is None:
            df[c] = df[c].astype("category")
        else:
            df[c] = pd.Categorical(df[c], categories=cat_categories[c])
    return df


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    tp = preprocess(train)
    cat_categories = {c: tp[c].cat.categories for c in CAT_COLS}
    y = tp[TARGET].values
    X = add_h1(train, tp.drop(columns=[TARGET]))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    models = []
    for tr, va in skf.split(X, y):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]),
                      num_boost_round=2000, valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        models.append(m)

    ths = np.arange(0.05, 0.95, 0.005)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    bi = int(np.argmax(f1s))
    BEST_TH = ths[bi]
    print(f"OOF AUC : {roc_auc_score(y, oof):.4f}")
    print(f"OOF AP  : {average_precision_score(y, oof):.4f}")
    print(f"OOF F1  : {f1s[bi]:.4f} @ th={BEST_TH:.3f}")

    tp_test = preprocess(test, cat_categories=cat_categories)
    tp_test = tp_test.drop(columns=[TARGET], errors="ignore")
    Xte = add_h1(test, tp_test)[X.columns]
    proba = np.mean([m.predict(Xte, num_iteration=m.best_iteration) for m in models], axis=0)
    label = (proba >= BEST_TH).astype(int)
    print(f"予測正例率: {label.mean():.4f} / 学習: {y.mean():.4f}")

    sample = pd.read_csv("data/sample_submit.csv", header=None, names=["企業ID", "購入フラグ"])
    pred_df = pd.DataFrame({"企業ID": test["企業ID"].values, "pred": label})
    sub = sample[["企業ID"]].merge(pred_df, on="企業ID", how="left")
    assert sub["pred"].notna().all(), "予測欠損あり"
    sub["pred"] = sub["pred"].astype(int)
    out = "submission/submission_h1.csv"
    sub.to_csv(out, index=False, header=False, lineterminator="\n")
    print(f"保存: {out} 行数={len(sub)} 正例={int(sub['pred'].sum())}")


if __name__ == "__main__":
    main()
