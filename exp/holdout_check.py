"""アンサンブルの上乗せが本物か、完全ホールドアウトで確かめる。

ensemble_experts.py はエキスパートOOFとメタ学習を同じ5分割の上で行うため、
メタの学習に使う Z[train] が「評価foldのラベルを見たモデル」から作られている
（リスタッキングのリーク）。ノイズエキスパートを混ぜる試験では検出できない
種類の楽観なので、ここで別途つぶす。

手順（1リピート）:
  1. train を A / B に層化分割する。B は最後まで一切触らない
  2. A の中だけで5分割し、各エキスパートのOOFを作る
  3. その A のOOF でメタの重みと閾値を決める
  4. 各エキスパートを A 全体で再学習し、B を予測する
  5. B 上で E0単体 / 等分ブレンド / メタ を比較する

B はどのエキスパートもメタも閾値決定も見ていないため、ここでの差は本物。

閾値則の比較（`--threshold-rules`）:
  exp020 は「順位付け(AUC/AP)は同等以上なのに閾値を当てられず F1 が 0/5」で
  REJECT された。閾値の決め方そのものが効いている可能性が高いので、同じ
  エキスパート予測に対して閾値則だけを差し替えて比較できるようにする。
  どの則も **A の情報だけ**で閾値を決める（rate は B のスコア分布だけを使い、
  B のラベルは見ない）。

  python exp/holdout_check.py [--n-reps 3] [--alpha 0.01]
                              [--threshold-rules oof_best fold_mean rate half_f1]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from organization_features import (  # noqa: E402
    ORG_SCORE_COL, build_org_model, fold_org_preds, load_org_text,
)
from text_features import TEXT_COL, build_model, nested_text_pred  # noqa: E402
from expert_groups import (  # noqa: E402
    e0_anchor_cols, e1_finance_cols, e2_survey_cols, e6_manual_cols,
)
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, EXPERTS, THS, _lgbm, _survey_lr, build_features, e4_inputs,
)
from meta_blend import (  # noqa: E402
    apply_meta, fit_meta, select_alpha, to_logit, uniform_blend,
)


def fit_predict_experts(X, y, txt, org, tr, target, seed, cols, e4=None):
    """tr で全エキスパートを学習し、target 行を予測する。

    E0 のテキスト特徴は tr の内側CVで作る（tr 内で完結＝リーク無し）。
    target 側は tr 全体で学習したテキストモデルで付ける。

    e4=(入力, モデル生成関数) を渡すと E4 の実装を差し替える（--org-embed）。
    E0 は常に org(TF-IDF) を使うので、差し替えの有無で E0 の予測は完全一致し、
    rep 単位の厳密なペア比較ができる。
    """
    c0, c1, c2, c6 = cols
    e4_in, e4_model = e4 if e4 is not None else (org, build_org_model)
    Xtr, Xta = X.iloc[tr], X.iloc[target]
    out = {}

    m_txt = build_model().fit(txt[tr], y[tr])
    out["E3_dx_text"] = m_txt.predict_proba(txt[target])[:, 1]
    m_org = e4_model().fit(e4_in[tr], y[tr])
    out["E4_org_text"] = m_org.predict_proba(e4_in[target])[:, 1]

    t_tr, t_ta = nested_text_pred(txt[tr], y[tr], txt[target])
    o_tr, o_ta, _ = fold_org_preds(org[tr], y[tr], org[target], org[:1])
    A_tr, A_ta = Xtr[c0].copy(), Xta[c0].copy()
    A_tr[DX_SCORE_COL], A_ta[DX_SCORE_COL] = t_tr, t_ta
    A_tr[ORG_SCORE_COL], A_ta[ORG_SCORE_COL] = o_tr, o_ta
    out["E0_anchor"], _ = _lgbm(A_tr, y[tr], A_ta, y[target], A_ta, seed)

    for name, c in (("E1_finance", c1), ("E6_manual", c6)):
        out[name], _ = _lgbm(Xtr[c], y[tr], Xta[c], y[target], Xta[c], seed)
    out["E2_survey"], _ = _survey_lr(Xtr[c2], y[tr], Xta[c2], Xta[c2])
    return out


SUBSETS = {
    "meta_6本": EXPERTS,
    "meta_4本": tuple(n for n in EXPERTS if n not in ("E3_dx_text", "E4_org_text")),
}


def _stack(preds, experts=EXPERTS):
    return to_logit(np.column_stack([preds[n] for n in experts]))


def _best_th(y, p):
    return THS[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in THS]))]


# 閾値則。すべて A の情報だけで決める（rate のみ B のスコア分布を使うがラベルは見ない）。
THRESHOLD_RULES = ("oof_best", "fold_mean", "rate", "half_f1")


def _thresholds(pA, yA, pB, folds, rules):
    """A の予測から閾値を決める。B のラベルは一切使わない。

    oof_best  A全体のOOFでF1最大（現行。argmaxを1点で取るので過学習しやすい）
    fold_mean A内5foldそれぞれのF1最大閾値の平均（平均化で分散を落とす）
    rate      A での予測正例率を B のスコア分位点に転写（rate matching）
    half_f1   達成F1の半分。校正済み確率のF1最適閾値は F1*/2 という既知の関係を
              使う。自由度が最小＝最も過学習しにくい
    """
    out = {}
    best = None
    if {"oof_best", "rate", "half_f1"} & set(rules):
        best = _best_th(yA, pA)
    for r in rules:
        if r == "oof_best":
            out[r] = float(best)
        elif r == "fold_mean":
            out[r] = float(np.mean([_best_th(yA[va], pA[va]) for _, va in folds]))
        elif r == "rate":
            rate = float((pA >= best).mean())
            out[r] = float(np.quantile(pB, 1 - rate)) if 0 < rate < 1 else float(best)
        elif r == "half_f1":
            out[r] = float(f1_score(yA, (pA >= best).astype(int)) / 2)
        else:
            raise ValueError(f"未知の閾値則: {r}")
    return out


def _score_at(y, p, th):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1_score(y, (p >= th).astype(int)))


def one_rep(X, y, txt, org, cols, rep, alpha, test_size, rules=("oof_best",),
            target="zero", e4=None):
    idx = np.arange(len(y))
    A, B = train_test_split(idx, test_size=test_size, stratify=y, random_state=rep)

    # 2. A の中だけでエキスパートOOFを作る
    oof = {n: np.zeros(len(A)) for n in EXPERTS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
    folds = list(skf.split(A, y[A]))
    for tr, va in folds:
        p = fit_predict_experts(X, y, txt, org, A[tr], A[va], rep, cols, e4)
        for n in EXPERTS:
            oof[n][va] = p[n]

    # 3. A のOOFで alpha と重みを決める（Bは一切見ない）
    yA = y[A]
    fitted, alphas = {}, {}
    for label, members in SUBSETS.items():
        Za = _stack(oof, members)
        a = (alpha if alpha is not None
             else select_alpha(Za, yA, seed=rep, target=target)[0])
        w, b = fit_meta(Za, yA, alpha=a, target=target)
        fitted[label] = (members, w, b)
        alphas[label] = a

    # 4. A全体で再学習 -> B を予測
    pb = fit_predict_experts(X, y, txt, org, A, B, rep, cols, e4)
    yB = y[B]

    # A側/B側のスコアを同じ形で並べる（閾値はAから、評価はBで）
    # エキスパート単体も入れる: 合成後に改善が消えるのが「E1自体が弱るから」なのか
    # 「メタが薄めるから」なのかを切り分けるため。
    pairs = {n: (oof[n], pb[n]) for n in EXPERTS}
    pairs["uniform_6本"] = (uniform_blend(_stack(oof, EXPERTS)),
                            uniform_blend(_stack(pb, EXPERTS)))
    for label, (members, w, b) in fitted.items():
        pairs[label] = (apply_meta(_stack(oof, members), w, b),
                        apply_meta(_stack(pb, members), w, b))

    # 5. 閾値則ごとに B を評価
    out = {}
    for label, (pA, pB) in pairs.items():
        for rule, t in _thresholds(pA, yA, pB, folds, rules).items():
            out[(label, rule)] = dict(_score_at(yB, pB, t), th=t,
                                      pos_rate=float((pB >= t).mean()))
    weights = {lab: dict(zip(mem, w)) for lab, (mem, w, _) in fitted.items()}
    return out, weights, alphas


def run(n_reps, alpha, test_size, finance_ratios=False, rules=("oof_best",),
        target="zero", survey_step=False, org_embed=False):
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, _ = build_features(train, test, finance_ratios=finance_ratios,
                             survey_step=survey_step)
    txt = train[TEXT_COL].fillna("").astype(str).values
    org = load_org_text(train)
    # E4 の入力/モデル。org_embed=False なら現行と完全一致（e4=None 相当）。
    e4_in, _, e4_model = e4_inputs(train, train, org_embed)
    e4 = (e4_in, e4_model) if org_embed else None
    cols = (e0_anchor_cols(X.columns),
            e1_finance_cols(with_ratios=finance_ratios),
            e2_survey_cols(with_step=survey_step), e6_manual_cols(X.columns))
    print(f"A={int(len(y) * (1 - test_size))}行 / B(ホールドアウト)={int(len(y) * test_size)}行"
          f" x {n_reps}リピート, alpha={alpha}, メタ縮小先={target}"
          f"{', E1にH21財務比4本' if finance_ratios else ''}"
          f"{', E2にH22閾値指標4本' if survey_step else ''}"
          f"{', E4を組織図embeddingに差し替え' if org_embed else ''}")

    recs, ws, alphas = [], [], []
    for rep in range(n_reps):
        print(f"  rep {rep + 1}/{n_reps} ...", flush=True)
        r, w, a = one_rep(X, y, txt, org, cols, rep, alpha, test_size, rules,
                          target, e4)
        ws.append(w)
        alphas.append(a)
        for (m, rule), v in r.items():
            recs.append(dict(rep=rep, model=m, rule=rule, **v))
    if alpha is None:
        print(f"\nA内で選ばれたalpha (rep別): {alphas}")

    df = pd.DataFrame(recs)
    ref = rules[0]

    # AUC/AP は閾値に依存しないので1度だけ出す
    print("\n=== ホールドアウトB上の順位系スコア (rep平均±std, 閾値非依存) ===")
    g = df[df.rule == ref].groupby("model")[["auc", "ap"]].agg(["mean", "std"])
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n=== F1 × 閾値則 (rep平均±std) ===")
    f1t = df.pivot_table(index="model", columns="rule", values="f1",
                         aggfunc=["mean", "std"])
    print(f1t.reindex(columns=rules, level=1).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\n=== 参考: 選ばれた閾値と B上の予測正例率 (rep平均) ===")
    print(df.pivot_table(index="model", columns="rule",
                         values=["th", "pos_rate"], aggfunc="mean").reindex(
        columns=rules, level=1).to_string(float_format=lambda v: f"{v:.3f}"))

    piv = df.pivot(index="rep", columns=["model", "rule"])
    others = [m for m in df.model.unique() if m != "E0_anchor"]
    print(f"\n=== E0単体との対リピート差分（閾値則={ref}）===")
    for k in ["auc", "ap", "f1"]:
        for m in others:
            d = piv[(k, m, ref)] - piv[(k, "E0_anchor", ref)]
            print(f"  Δ{k.upper():3s} {m:12s}: mean {d.mean():+.4f}  "
                  f"正 {int((d > 0).sum())}/{len(d)}  各rep {np.round(d.values, 4)}")

    print(f"\n=== 閾値則の対リピート差分（各モデル内、基準={ref}）===")
    for m in df.model.unique():
        for r in rules[1:]:
            d = piv[("f1", m, r)] - piv[("f1", m, ref)]
            print(f"  ΔF1 {m:12s} {r:9s}: mean {d.mean():+.4f}  "
                  f"正 {int((d > 0).sum())}/{len(d)}  各rep {np.round(d.values, 4)}")

    print("\n=== meta_6本 との対リピート差分（4本に落とした効果、閾値則ごと）===")
    for r in rules:
        d = piv[("f1", "meta_4本", r)] - piv[("f1", "meta_6本", r)]
        print(f"  ΔF1 {r:9s}: mean {d.mean():+.4f}  正 {int((d > 0).sum())}/{len(d)}"
              f"  各rep {np.round(d.values, 4)}")

    print("\n=== メタの重み (rep平均, 正規化%) ===")
    for label in SUBSETS:
        vals = pd.DataFrame([w[label] for w in ws])
        norm = vals.div(vals.sum(axis=1), axis=0) * 100
        print(f"  [{label}]  ばらつき(rep間std)も併記")
        for n in norm.mean().sort_values(ascending=False).index:
            print(f"    {n:12s} {norm[n].mean():5.1f}%  ±{norm[n].std():4.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--alpha", type=float, default=None,
                   help="省略時は A の中だけで自動選択")
    p.add_argument("--test-size", type=float, default=0.35)
    p.add_argument("--finance-ratios", action="store_true",
                   help="E1 に H21 財務比4本を足す（E0は現行94列のまま）")
    p.add_argument("--survey-step", action="store_true",
                   help="E2 に H22 閾値指標4本を足す（E0は現行94列のまま）")
    p.add_argument("--org-embed", action="store_true",
                   help="E4 を組織図の OpenAI embedding+LR に差し替える（E0は不変）")
    p.add_argument("--threshold-rules", nargs="+", default=["oof_best"],
                   choices=list(THRESHOLD_RULES),
                   help="比較する閾値則。先頭が差分の基準（既定は現行のoof_bestのみ）")
    p.add_argument("--meta-target", default="zero", choices=["zero", "uniform"],
                   help="メタ重みの縮小先。uniform は等分ブレンドへ縮む")
    a = p.parse_args()
    run(a.n_reps, a.alpha, a.test_size, a.finance_ratios,
        tuple(a.threshold_rules), a.meta_target, a.survey_step, a.org_embed)
