"""H22: E2アンケートエキスパート単体の変種比較（合成前の足切り）。

E2 は LR + 12列なので1seedが一瞬で回る。合成(ensemble_experts)やホールドアウトに
かける前に、**E2単体で 10〜20seed の同一seedペア比較**を済ませて候補を絞る。
却下則3（構成員を増やすと重み推定ノイズが漏れる）は今回は無関係
（エキスパートは増やさず E2 の列だけ変える）が、E2 が劣化すれば重み最大の
構成員が劣化するので、ここでの判定はそのまま合成に効く。

変種:
  S0_現行        アンケート11本 + アンケート７_isna の12列（exp019 と同一）
  S1_a7中央値埋め S0 の a7 の 0埋めを外側train内の観測中央値埋めに置換
  shape/step/gap  S0 + 各群（`survey_features.SURVEY_GROUPS`）
  all             S0 + 全群
  接尾辞 `+fill`  同じ構成に S1 の中央値埋めを併用

  python exp/compare_survey.py [--n-seeds 20] [--variants S0 shape gap ...]

出力: exp/_h22_survey_scores.csv（変種×seedの生スコア）
      exp/_h22_survey_summary.csv（S0 に対する判定）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import TARGET, preprocess  # noqa: E402
from decision import format_report, verdict  # noqa: E402
from expert_groups import e2_survey_cols  # noqa: E402
from survey_features import (  # noqa: E402
    SURVEY_GROUPS, a7_median_fill, add_survey_features, survey_feature_report,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
SURVEY_LR = dict(C=1.0, max_iter=2000, random_state=0)

# (変種名, 追加する群, a7中央値埋めするか)
VARIANTS = {
    "S0_現行": ((), False),
    "S1_fill": ((), True),
    "shape": (("shape",), False),
    "step": (("step",), False),
    "gap": (("gap",), False),
    "all": (("shape", "step", "gap"), False),
    "shape+fill": (("shape",), True),
    "step+fill": (("step",), True),
    "gap+fill": (("gap",), True),
    "all+fill": (("shape", "step", "gap"), True),
}
BASE = "S0_現行"


def _oof_scores(X, y, cols, seed, fill):
    """E2-LR の OOF を1seed分作り、指標を返す。"""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if fill:
            Xtr, Xva = a7_median_fill(Xtr, Xva)
        model = make_pipeline(StandardScaler(), LogisticRegression(**SURVEY_LR))
        model.fit(Xtr[cols].fillna(0), y[tr])
        oof[va] = model.predict_proba(Xva[cols].fillna(0))[:, 1]
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in THS]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, oof), ap=average_precision_score(y, oof),
                f1=f1s[b], th=THS[b])


def run(n_seeds, names):
    train = pd.read_csv("data/train.csv")
    tp = preprocess(train)
    y = tp[TARGET].values
    X = add_survey_features(train, tp.drop(columns=[TARGET]))
    base_cols = e2_survey_cols()

    print("=== 派生列の事前診断（単体AUC/AP と既存12列との最大|r|）===")
    rep = survey_feature_report(train, y)
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n正例率 {y.mean():.4f} / train {len(y)}行 / {n_seeds}seed\n")

    seeds = list(range(n_seeds))
    tables, recs = {}, []
    for name in names:
        groups, fill = VARIANTS[name]
        cols = base_cols + [c for g in groups for c in SURVEY_GROUPS[g]]
        missing = [c for c in cols if c not in X.columns]
        assert not missing, f"未知の列: {missing}"
        df = pd.DataFrame([_oof_scores(X, y, cols, s, fill) for s in seeds])
        df.attrs["n_feat"] = len(cols)
        tables[name] = df
        print(f"[{name}] n_feat={len(cols)}  "
              f"AP {df.ap.mean():.4f}±{df.ap.std():.4f}  "
              f"AUC {df.auc.mean():.4f}±{df.auc.std():.4f}  "
              f"F1 {df.f1.mean():.4f}±{df.f1.std():.4f}")
        recs += [dict(variant=name, seed=s, n_feat=len(cols), **r)
                 for s, r in zip(seeds, df.to_dict("records"))]

    pd.DataFrame(recs).to_csv(os.path.join(OUT_DIR, "_h22_survey_scores.csv"),
                              index=False)

    print(f"\n=== {BASE} に対する同一seedペア比較 ===")
    rows = []
    for name in names:
        if name == BASE:
            continue
        v, d = verdict(tables[BASE], tables[name])
        print("\n" + format_report(name, f"E2 + {name}", tables[BASE],
                                   tables[name], v, d))
        rows.append(dict(variant=name, verdict=v,
                         **{f"d{k}_{s}": d[k][s] for k in ("ap", "auc", "f1")
                            for s in ("mean", "n_pos")}))
    out = os.path.join(OUT_DIR, "_h22_survey_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n保存: {OUT_DIR}/_h22_survey_{{scores,summary}}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS))
    a = p.parse_args()
    names = [n for n in VARIANTS if n in a.variants]
    if BASE not in names:
        names = [BASE] + names
    run(a.n_seeds, names)
