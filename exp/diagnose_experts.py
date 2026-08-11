"""エキスパート分割の事前診断（アンサンブルを組む前の足切り）。

exp012 では LGBM+CatBoost の rank平均が rank相関 0.9495 ＝ 多様性ほぼゼロで、
OOF AUC/AP は改善したのに Public は 0.7500 -> 0.7241 と沈んだ。同じ轍を踏まない
ため、メタ学習器を作り込む前に「各エキスパートが単体でどれだけ効くか」と
「エキスパート同士がどれだけ相関しているか」だけを先に測る。

  python exp/diagnose_experts.py [--seed 42]

出力:
  exp/_expert_diag_scores.csv   単体スコア
  exp/_expert_diag_corr.csv     OOF予測のSpearman相関行列
  exp/_expert_diag_oof.npz      OOF予測（後段のメタ学習器の設計に再利用）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import BASE_PARAMS, TARGET, preprocess  # noqa: E402
from make_submission_h1_h8 import add_h1  # noqa: E402
from organization_features import (  # noqa: E402
    ORG_SCORE_COL,
    add_org_manual_features,
    build_org_model,
    fold_org_preds,
    load_org_text,
)
from dx_outlook_features import add_dx_outlook_manual_features  # noqa: E402
from company_overview_features import add_company_overview_six_rate_flags  # noqa: E402
from text_features import TEXT_COL, build_model, nested_text_pred  # noqa: E402
from expert_groups import e1_finance_cols, e2_survey_cols, e6_manual_cols  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
# 診断のロジスティック回帰。742行に対し列が少ないので素のL2で十分。
# random_state を明示するのは、liblinear のシャッフル由来の実行ごとのブレを
# 診断に持ち込まないため（本番の LR_PARAMS 側は未指定のまま＝別途要修正）。
SURVEY_LR = dict(C=1.0, max_iter=2000, random_state=0)


def _build_full_features(train):
    """現行 exp017 と同じ 92列（fold内スコア2本を除く）を組む。"""
    tp = preprocess(train)
    y = tp[TARGET].values
    X = add_h1(train, tp.drop(columns=[TARGET]))
    X = add_org_manual_features(train, X)
    X = add_dx_outlook_manual_features(train, X)
    X = add_company_overview_six_rate_flags(train, X)
    return X, y


def _lgbm_pred(Xtr, ytr, Xva, yva, seed):
    """early stopping を valid で行う。harness.run_cv と同一手順。"""
    params = {**BASE_PARAMS, "seed": seed}
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=2000,
                  valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(Xva, num_iteration=m.best_iteration)


def _lr_pred(Xtr, ytr, Xva):
    model = make_pipeline(StandardScaler(), LogisticRegression(**SURVEY_LR))
    model.fit(Xtr.fillna(0), ytr)
    return model.predict_proba(Xva.fillna(0))[:, 1]


def _scores(y, oof):
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in THS]
    best = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, oof), ap=average_precision_score(y, oof),
                f1=f1s[best], th=THS[best])


def run(seed):
    train = pd.read_csv("data/train.csv")
    X, y = _build_full_features(train)
    txt = train[TEXT_COL].fillna("").astype(str).values
    org = load_org_text(train)

    c1, c2 = e1_finance_cols(), e2_survey_cols()
    c6 = e6_manual_cols(X.columns)
    missing = [c for c in c1 + c2 + c6 if c not in X.columns]
    assert not missing, f"未知の列: {missing}"
    print(f"E0 全体      : {X.shape[1] + 2} 列 (fold内スコア2本を含む)")
    print(f"E1 財務・規模 : {len(c1)} 列")
    print(f"E2 アンケート : {len(c2)} 列")
    print(f"E6 辞書・構造 : {len(c6)} 列")
    covered = set(c1) | set(c2) | set(c6)
    print(f"未割当       : {sorted(set(X.columns) - covered)}")

    names = ["E0_anchor", "E1_finance", "E2_survey_lr", "E2_survey_lgbm",
             "E3_dx_text", "E4_org_text", "E6_manual"]
    oof = {n: np.zeros(len(X)) for n in names}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"  fold {k}/5 ...", flush=True)
        Xtr, Xva = X.iloc[tr], X.iloc[va]

        # E3 / E4: 外側trainでfitして外側validを予測（単体エキスパートなので内側CV不要）
        m_txt = build_model().fit(txt[tr], y[tr])
        oof["E3_dx_text"][va] = m_txt.predict_proba(txt[va])[:, 1]
        m_org = build_org_model().fit(org[tr], y[tr])
        oof["E4_org_text"][va] = m_org.predict_proba(org[va])[:, 1]

        # E0: 現行構成。テキストスコアは内側CVで作って特徴に足す
        t_tr, t_va = nested_text_pred(txt[tr], y[tr], txt[va])
        o_tr, o_va, _ = fold_org_preds(org[tr], y[tr], org[va], org[:1])
        A_tr, A_va = Xtr.copy(), Xva.copy()
        A_tr["DX展望_購入確率"], A_va["DX展望_購入確率"] = t_tr, t_va
        A_tr[ORG_SCORE_COL], A_va[ORG_SCORE_COL] = o_tr, o_va
        oof["E0_anchor"][va] = _lgbm_pred(A_tr, y[tr], A_va, y[va], seed)

        oof["E1_finance"][va] = _lgbm_pred(Xtr[c1], y[tr], Xva[c1], y[va], seed)
        oof["E6_manual"][va] = _lgbm_pred(Xtr[c6], y[tr], Xva[c6], y[va], seed)
        oof["E2_survey_lgbm"][va] = _lgbm_pred(Xtr[c2], y[tr], Xva[c2], y[va], seed)
        oof["E2_survey_lr"][va] = _lr_pred(Xtr[c2], y[tr], Xva[c2])

    table = pd.DataFrame({n: _scores(y, oof[n]) for n in names}).T
    table.insert(0, "n_feat", [X.shape[1] + 2, len(c1), len(c2), len(c2), 1, 1, len(c6)])
    print("\n=== 単体OOFスコア (seed=%d) ===" % seed)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    P = np.column_stack([oof[n] for n in names])
    corr = pd.DataFrame(spearmanr(P).correlation, index=names, columns=names)
    print("\n=== OOF予測のSpearman相関 ===")
    print(corr.to_string(float_format=lambda v: f"{v:.3f}"))

    table.to_csv(os.path.join(OUT_DIR, "_expert_diag_scores.csv"))
    corr.to_csv(os.path.join(OUT_DIR, "_expert_diag_corr.csv"))
    np.savez(os.path.join(OUT_DIR, "_expert_diag_oof.npz"), y=y,
             **{n: oof[n] for n in names})
    print(f"\n保存: {OUT_DIR}/_expert_diag_{{scores,corr,oof}}.*")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args().seed)
