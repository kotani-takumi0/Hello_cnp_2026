"""H24: E2アンケートエキスパートの**モデル族**を替える（合成前の足切り）。

exp019 以降の6本は LGBM×3 + LR×3 で、仮説空間が「軸平行の階段関数」と
「特徴空間の超平面」の2つしかない。**滑らかで軸に平行でない決定境界**は
どの構成員も表現していない。それを入れるならどこか、を測る。

E2 を対象にする理由（`_expert_diag_scores.csv` / `_expert_diag_corr.csv` より）:
  - 同じ12列で LGBM 0.6446 < LR 0.6806。**木がこの12列で実際に負けている**
  - 94列を食う E0 との相関が 0.187 しかない ＝ 大きい方の木も取りこぼしている
  - 合成重みが常に最大（38〜42%）。他と |r|<0.07 なので非負重みが厚く載る

exp025(H22) が落ちたのは「閾値で切った列」＝木が自分で作れる形だったから。
モデル族の差し替えは失敗の型が違う、というのがこの実験の賭け。

**事前登録した停止条件**: 候補の OOF が現行 M0_lr と Spearman 0.90 以上なら、
単体スコアが何であれ却下する。exp012(CatBoost, r=0.9495) と exp021(H9, r=0.927)
の2件が同じ形で落ちており、相関が高いものは合成に寄与しないと分かっている。

  python exp/compare_survey_kernel.py [--n-seeds 15] [--models M0_lr M1_svm ...]

出力: exp/_h24_kernel_scores.csv  変種×seedの生スコア
      exp/_h24_kernel_summary.csv M0_lr に対する判定＋相関ゲート
      exp/_h24_kernel_oof.npz     OOF予測（合成の設計に再利用）
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import TARGET, preprocess  # noqa: E402
from decision import format_report, verdict  # noqa: E402
from expert_groups import e2_survey_cols  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)

# 現行 E2 と完全に同一（ensemble_experts.SURVEY_LR）。random_state は seed ではなく
# 0 固定＝本番と同じ。ここを seed 連動にすると本番の再現にならない。
SURVEY_LR = dict(C=1.0, max_iter=2000, random_state=0)

# RBF-SVM のハイパラは**外側trainの中だけ**で内側3分割CVから選ぶ。
# 選択基準は AP（decision_function のランクだけで決まるので確率化は不要＝速い）。
#
# `class_weight="balanced"` は必須。正例率24%のまま素の hinge 損失を最小化すると
# マージンが多数派側に寄って OOF AUC 0.5546 まで落ちる（診断実測）。balanced に
# するだけで 0.6547 まで戻る。LR がこれ無しで平気なのは対数損失が確率を返すためで、
# SVM に同じ扱いをするのは比較として不当。
SVM_GRID = {"svc__C": [0.3, 1.0, 3.0, 10.0],
            "svc__gamma": [0.003, 0.01, 0.03, "scale", 0.3]}
SVM_FIXED = dict(kernel="rbf", class_weight="balanced")
INNER_CV_SEED = 0  # 内側分割は seed 非連動。ペア比較の変動源を外側分割だけに絞る

BASE = "M0_lr"


def fp_lr(Xtr, ytr, Xva):
    """現行 E2。StandardScaler + L2ロジスティック回帰。"""
    m = make_pipeline(StandardScaler(), LogisticRegression(**SURVEY_LR))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def fp_svm(Xtr, ytr, Xva):
    """RBF-SVM。内側CVで (C, gamma) を選んでから Platt 確率化で refit。

    `probability=True` の内部CVはシャッフルするので random_state を必ず固定する
    （LRの liblinear で同じ穴を踏んでいる。experiments.md「再現性の修正」参照）。
    """
    base = Pipeline([("sc", StandardScaler()), ("svc", SVC(**SVM_FIXED))])
    gs = GridSearchCV(
        base, SVM_GRID, scoring="average_precision", n_jobs=-1,
        cv=StratifiedKFold(3, shuffle=True, random_state=INNER_CV_SEED))
    gs.fit(Xtr, ytr)
    best = {k.split("__")[1]: v for k, v in gs.best_params_.items()}
    m = Pipeline([("sc", StandardScaler()),
                  ("svc", SVC(probability=True, random_state=0,
                              **SVM_FIXED, **best))])
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def _gp(kernel, restarts):
    return make_pipeline(
        StandardScaler(),
        GaussianProcessClassifier(kernel=kernel, random_state=0,
                                  n_restarts_optimizer=restarts))


def fp_gp(Xtr, ytr, Xva):
    """ガウス過程分類（等方RBF）。カーネル長は周辺尤度で決まるので外から与えない。

    Laplace近似の確率をそのまま使う（SVMと違い後付けの校正が要らない）。
    **この変種はこの実験で最も情報量のある結果を出す**: 学習された length_scale が
    データ直径に比べて十分大きければ、RBFはデータ全域でほぼ一定＝実質線形になり、
    「12列に滑らかな非線形構造が無い」ことの積極的な証拠になる。
    """
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2)))
    m = _gp(kernel, 1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def fp_gp_ard(Xtr, ytr, Xva):
    """ガウス過程分類（ARD = 列ごとに長さスケール）。等方版の弱点を潰した版。

    等方RBFは12列を等距離に扱うので、効く列と効かない列が混ざっていると
    無関係な列の分散に埋もれて不利になる。ARD なら周辺尤度が列ごとに長さを
    決めるので、「非線形が無い」という結論を出す前にこちらでも確認する。
    ハイパラが1個から12個に増えるぶん n=742 では過学習しうるのが引き換え。
    """
    n = Xtr.shape[1]
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=np.ones(n), length_scale_bounds=(1e-1, 1e3)))
    m = _gp(kernel, 0)  # 12次元の再スタートは高価。初期値1点で回す
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


MODELS = {"M0_lr": fp_lr, "M1_svm": fp_svm, "M2_gp": fp_gp,
          "M3_gp_ard": fp_gp_ard}


def oof_predict(fit_predict, X, y, seed):
    """1 seed 分の OOF を作る。分割は本番(ensemble_experts)と同一の切り方。"""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        oof[va] = fit_predict(X.iloc[tr], y[tr], X.iloc[va])
    return oof


def score(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1s[b], th=THS[b])


def run(n_seeds, names, gate):
    train = pd.read_csv("data/train.csv")
    tp = preprocess(train)
    y = tp[TARGET].values
    cols = e2_survey_cols()
    X = tp[cols].fillna(0)  # 本番 `_survey_lr` と同じ 0 埋め

    print(f"E2 アンケート {len(cols)}列 / train {len(y)}行 / 正例率 {y.mean():.4f}"
          f" / {n_seeds}seed")
    print(f"相関ゲート: Spearman >= {gate:.2f} なら却下（事前登録）\n")

    seeds = list(range(n_seeds))
    tables, oofs, recs = {}, {}, []
    for name in names:
        t0 = time.time()
        arrs = [oof_predict(MODELS[name], X, y, s) for s in seeds]
        df = pd.DataFrame([score(y, p) for p in arrs])
        df.attrs["n_feat"] = len(cols)
        tables[name], oofs[name] = df, np.array(arrs)
        print(f"[{name}] AP {df.ap.mean():.4f}±{df.ap.std():.4f}  "
              f"AUC {df.auc.mean():.4f}±{df.auc.std():.4f}  "
              f"F1 {df.f1.mean():.4f}±{df.f1.std():.4f}  "
              f"({time.time() - t0:.1f}s)")
        recs += [dict(model=name, seed=s, **r)
                 for s, r in zip(seeds, df.to_dict("records"))]

    pd.DataFrame(recs).to_csv(
        os.path.join(OUT_DIR, "_h24_kernel_scores.csv"), index=False)
    np.savez(os.path.join(OUT_DIR, "_h24_kernel_oof.npz"), y=y,
             seeds=np.array(seeds), **oofs)

    print(f"\n=== 相関ゲート（{BASE} との Spearman、seed毎に計算して平均）===")
    corr = {}
    for name in names:
        if name == BASE:
            continue
        rs = [spearmanr(oofs[BASE][i], oofs[name][i]).statistic
              for i in range(n_seeds)]
        corr[name] = float(np.mean(rs))
        ok = "PASS" if corr[name] < gate else "GATE-FAIL (却下)"
        print(f"  {BASE} vs {name:8s}  r = {corr[name]:.4f}  "
              f"(min {min(rs):.4f} / max {max(rs):.4f})  -> {ok}")

    print(f"\n=== {BASE} に対する同一seedペア比較 ===")
    rows = []
    for name in names:
        if name == BASE:
            continue
        v, d = verdict(tables[BASE], tables[name])
        print("\n" + format_report(name, f"E2 を {name} に差し替え",
                                   tables[BASE], tables[name], v, d))
        print(f"  相関 => r={corr[name]:.4f} "
              f"{'(ゲート通過)' if corr[name] < gate else '(ゲート違反→却下)'}")
        rows.append(dict(model=name, verdict=v, spearman=corr[name],
                         gate_pass=bool(corr[name] < gate),
                         **{f"d{k}_{s}": d[k][s] for k in ("ap", "auc", "f1")
                            for s in ("mean", "n_pos")}))
    out = os.path.join(OUT_DIR, "_h24_kernel_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n保存: {OUT_DIR}/_h24_kernel_{{scores,summary}}.csv, _h24_kernel_oof.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=15)
    p.add_argument("--models", nargs="+", default=list(MODELS),
                   choices=list(MODELS))
    p.add_argument("--gate", type=float, default=0.90,
                   help="Spearman がこれ以上なら却下（事前登録した停止条件）")
    a = p.parse_args()
    names = [n for n in MODELS if n in a.models]
    if BASE not in names:
        names = [BASE] + names
    run(a.n_seeds, names, a.gate)
