"""baseline + H1(利益率3本) にテキスト系仮説を足して提出ファイルを作る。

exp002(make_submission_h1.py)と同一手順（SEED=42, 5fold, OOF閾値, fold平均）に
テキスト特徴のみ追加。テキスト特徴はfold内で完結させる:
  - fold train部分 → 内側5foldのOOF予測（自分の正解を見ない）
  - fold valid部分 → fold train全体で学習したテキストモデルの予測
  - test          → 同じくfold train全体で学習したモデルの予測（fold毎に作り直す）
これによりテキストモデルが見る正解は常にそのfoldのtrainに限定される。

H9(LLM3軸)はLLMが正解を見ずに付けた行単位スコアなのでfold分割不要。

  python exp/make_submission_h1_h8.py                  # exp004 H8    (今後のDX展望のみ)
  python exp/make_submission_h1_h8.py --overview       # exp005 H8b   (企業概要+今後のDX展望)
  python exp/make_submission_h1_h8.py --llm            # exp006 H8+H9 (H8にLLM3軸を上乗せ)
  python exp/make_submission_h1_h8.py --llm --no-text  # exp007 H9    (TF-IDFをLLM3軸で置換)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_features import (
    TEXT_COL,
    OVERVIEW_COL,
    build_model,
    fold_overview_preds,
    tokenize_overview_texts,
    INNER_SPLITS,
    INNER_SEED,
)
from llm_features import add_llm_axes, TRAIN_LLM, TEST_LLM
from dx_outlook_features import add_dx_outlook_manual_features
from company_overview_features import (
    add_company_overview_manual_features,
    add_company_overview_top_flags,
    add_company_overview_six_rate_flags,
)
from organization_features import (
    ORG_SCORE_COL,
    add_org_manual_features,
    fold_org_preds,
    load_org_text,
)

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


def join_text(df, cols):
    s = df[cols[0]].fillna("")
    for c in cols[1:]:
        s = s + " " + df[c].fillna("")
    return s.values


def fold_text_preds(txt_tr, y_tr, txt_va, txt_te):
    """fold train のみで学習し (train部分OOF, valid予測, test予測) を返す。"""
    oof = np.zeros(len(txt_tr))
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    for i, j in inner.split(txt_tr, y_tr):
        m = build_model().fit(txt_tr[i], y_tr[i])
        oof[j] = m.predict_proba(txt_tr[j])[:, 1]
    full = build_model().fit(txt_tr, y_tr)
    return oof, full.predict_proba(txt_va)[:, 1], full.predict_proba(txt_te)[:, 1]


def resolve_tag(use_overview, use_llm, no_text, use_org_chart,
                use_dx_outlook_manual, use_overview_word_stack,
                use_company_overview_manual, overview_top_flags,
                use_overview_six_flags):
    """構成 -> 提出ファイルのタグ。既存 h1_h8 / h1_h8b は不変に保つ。"""
    text = "" if no_text else ("h8b" if use_overview else "h8")
    parts = ["h1"] + ([text] if text else []) + (["h9"] if use_llm else [])
    if use_org_chart:
        parts.append("h13")
    if use_dx_outlook_manual:
        parts.append("h14")
    if use_overview_word_stack:
        parts.append("h15")
    if use_company_overview_manual:
        parts.append("h16")
    if overview_top_flags:
        parts.append(f"h17top{overview_top_flags}")
    if use_overview_six_flags:
        parts.append("h18six")
    return "_".join(parts)


def main(use_overview, use_llm, no_text, use_org_chart, use_dx_outlook_manual,
         use_overview_word_stack, use_company_overview_manual,
         overview_top_flags, use_overview_six_flags):
    cols = (OVERVIEW_COL, TEXT_COL) if use_overview else (TEXT_COL,)
    feat_name = "テキスト_購入確率" if use_overview else "DX展望_購入確率"
    tag = resolve_tag(use_overview, use_llm, no_text, use_org_chart,
                      use_dx_outlook_manual, use_overview_word_stack,
                      use_company_overview_manual, overview_top_flags,
                      use_overview_six_flags)
    if no_text:
        print("テキストスタッキング: 使わない (--no-text)")
    else:
        print(f"テキスト入力: {cols} -> 特徴名 {feat_name}")
    print(f"LLM3軸(H9): {'使う' if use_llm else '使わない'}")
    print(f"組織図特徴(H13): {'使う' if use_org_chart else '使わない'}")
    print(f"DX展望明示特徴(H14): {'使う' if use_dx_outlook_manual else '使わない'}")
    print(f"企業概要wordスタック(H15): {'使う' if use_overview_word_stack else '使わない'}")
    print(f"企業概要明示特徴(H16): {'使う' if use_company_overview_manual else '使わない'}")
    print(f"企業概要上位フラグ(H17): {overview_top_flags or '使わない'}")
    print(f"企業概要高低6フラグ(H18): {'使う' if use_overview_six_flags else '使わない'}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    tp = preprocess(train)
    cat_categories = {c: tp[c].cat.categories for c in CAT_COLS}
    y = tp[TARGET].values
    X = add_h1(train, tp.drop(columns=[TARGET]))

    tp_test = preprocess(test, cat_categories=cat_categories).drop(columns=[TARGET], errors="ignore")
    Xte_base = add_h1(test, tp_test)

    if use_org_chart:
        X = add_org_manual_features(train, X)
        Xte_base = add_org_manual_features(test, Xte_base)

    if use_dx_outlook_manual:
        X = add_dx_outlook_manual_features(train, X)
        Xte_base = add_dx_outlook_manual_features(test, Xte_base)

    if use_company_overview_manual:
        X = add_company_overview_manual_features(train, X)
        Xte_base = add_company_overview_manual_features(test, Xte_base)

    if overview_top_flags:
        X = add_company_overview_top_flags(train, X, overview_top_flags)
        Xte_base = add_company_overview_top_flags(test, Xte_base, overview_top_flags)

    if use_overview_six_flags:
        X = add_company_overview_six_rate_flags(train, X)
        Xte_base = add_company_overview_six_rate_flags(test, Xte_base)

    # H9: train/test 双方で同じ位置に足すことで、下の [X.columns] で列順が揃う
    if use_llm:
        X = add_llm_axes(train, X, TRAIN_LLM)
        Xte_base = add_llm_axes(test, Xte_base, TEST_LLM)
    Xte_base = Xte_base[X.columns]

    txt_all = join_text(train, cols)
    txt_te = join_text(test, cols)
    overview_all = tokenize_overview_texts(join_text(train, (OVERVIEW_COL,)))
    overview_te = tokenize_overview_texts(join_text(test, (OVERVIEW_COL,)))
    org_all = load_org_text(train) if use_org_chart else None
    org_te = load_org_text(test) if use_org_chart else None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_preds = []
    gains = []
    for tr, va in skf.split(X, y):
        Xtr, Xva, Xte = X.iloc[tr].copy(), X.iloc[va].copy(), Xte_base.copy()
        if not no_text:
            t_tr, t_va, t_te = fold_text_preds(txt_all[tr], y[tr], txt_all[va], txt_te)
            Xtr[feat_name], Xva[feat_name], Xte[feat_name] = t_tr, t_va, t_te
        if use_org_chart:
            o_tr, o_va, o_te = fold_org_preds(org_all[tr], y[tr], org_all[va], org_te)
            Xtr[ORG_SCORE_COL], Xva[ORG_SCORE_COL], Xte[ORG_SCORE_COL] = o_tr, o_va, o_te
        if use_overview_word_stack:
            v_tr, v_va, v_te = fold_overview_preds(
                overview_all[tr], y[tr], overview_all[va], overview_te
            )
            Xtr["企業概要_購入確率"] = v_tr
            Xva["企業概要_購入確率"] = v_va
            Xte["企業概要_購入確率"] = v_te

        m = lgb.train(PARAMS, lgb.Dataset(Xtr, y[tr]), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(Xva, y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(Xva, num_iteration=m.best_iteration)
        test_preds.append(m.predict(Xte, num_iteration=m.best_iteration))
        gains.append(pd.Series(m.feature_importance("gain"), index=m.feature_name()))

    ths = np.arange(0.05, 0.95, 0.005)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    bi = int(np.argmax(f1s))
    best_th = ths[bi]
    print(f"OOF AUC : {roc_auc_score(y, oof):.4f}")
    print(f"OOF AP  : {average_precision_score(y, oof):.4f}")
    print(f"OOF F1  : {f1s[bi]:.4f} @ th={best_th:.3f}")

    imp = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
    print("重要度TOP10(gain, 5fold平均):")
    for i, (k, v) in enumerate(imp.head(10).items(), 1):
        print(f"  {i:2d}. {k}: {v:.1f}")

    proba = np.mean(test_preds, axis=0)
    label = (proba >= best_th).astype(int)
    print(f"予測正例率: {label.mean():.4f} / 学習: {y.mean():.4f}")

    sample = pd.read_csv("data/sample_submit.csv", header=None, names=["企業ID", "購入フラグ"])
    pred_df = pd.DataFrame({"企業ID": test["企業ID"].values, "pred": label})
    sub = sample[["企業ID"]].merge(pred_df, on="企業ID", how="left")
    assert sub["pred"].notna().all(), "予測欠損あり"
    sub["pred"] = sub["pred"].astype(int)
    out = f"submission/submission_{tag}.csv"
    sub.to_csv(out, index=False, header=False, lineterminator="\n")
    print(f"保存: {out} 行数={len(sub)} 正例={int(sub['pred'].sum())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--overview", action="store_true", help="企業概要も結合する(H8b)")
    p.add_argument("--llm", action="store_true", help="LLM3軸を足す(H9)")
    p.add_argument("--no-text", action="store_true",
                   help="テキストスタッキングを使わない(H8を外す)")
    p.add_argument("--org-chart", action="store_true",
                   help="組織図の明示特徴とTF-IDFスタッキング確率を足す(H13)")
    p.add_argument("--dx-outlook-manual", action="store_true",
                   help="DX展望の慎重/拡大/教育/未来文脈の明示特徴を足す(H14)")
    p.add_argument("--overview-word-stack", action="store_true",
                   help="企業概要word TF-IDF+LRスタッキング確率を別特徴として足す(H15)")
    p.add_argument("--company-overview-manual", action="store_true",
                   help="企業概要の業態/IT親和性/低購入業態の明示特徴を足す(H16)")
    p.add_argument("--overview-top-flags", type=int, choices=[1, 2, 3], default=0,
                   help="企業概要の高購入率上位キーワード群を上からn個だけ足す(H17)")
    p.add_argument("--overview-six-rate-flags", action="store_true",
                   help="企業概要の高購入率上位3群+低購入率上位3群を足す(H18)")
    a = p.parse_args()
    if a.no_text and a.overview:
        p.error("--no-text と --overview は同時に指定できない")
    if a.no_text and not a.llm:
        p.error("--no-text だけでは exp002(make_submission_h1.py) と同じになる")
    main(a.overview, a.llm, a.no_text, a.org_chart, a.dx_outlook_manual,
         a.overview_word_stack, a.company_overview_manual,
         a.overview_top_flags, a.overview_six_rate_flags)
