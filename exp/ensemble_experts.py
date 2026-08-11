"""特徴量グループごとのエキスパートを立て、重み付きで合成する。

現行 exp017 は94列を1つのLightGBMに入れている。診断(diagnose_experts.py)で
  - E2アンケートは他の全エキスパートと無相関(|r|<0.07)なのにAUC .68 ある
  - なのに E0 と E2 の相関は .187 しかなく、E0 はアンケートを取りこぼしている
ことが分かったため、グループを分けて合成する。

構成:
  E0 anchor  現行94列 LightGBM      … 交互作用を保持するアンカー。下振れ防止
  E1 finance 財務・規模32列 LightGBM
  E2 survey  アンケート12列 LR      … 順序尺度なので線形（診断でLGBMに明確勝ち）
  E3 dx_text 今後のDX展望 TF-IDF+LR
  E4 org_text 組織図 TF-IDF+LR
  E6 manual  辞書・構造48列 LightGBM

エキスパートのOOFは1つの外側5分割で作り、メタ学習器のみ外側fold内でfitする。
E0が内部にネストCVのテキストスタッキングを持つため、完全な3層ネストは
計算量が現実的でない。残る楽観は非負・L2制約下の重み6本分に限られる。

  python exp/ensemble_experts.py --seed 42 [--alpha 1.0] [--submit]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import BASE_PARAMS, TARGET  # noqa: E402
from harness import preprocess as harness_preprocess  # noqa: E402
from make_submission_h1_h8 import add_h1, preprocess as sub_preprocess, CAT_COLS  # noqa: E402
from organization_features import (  # noqa: E402
    ORG_SCORE_COL, add_org_manual_features, build_org_model,
    fold_org_preds, load_org_text,
)
from dx_outlook_features import add_dx_outlook_manual_features  # noqa: E402
from company_overview_features import add_company_overview_six_rate_flags  # noqa: E402
from finance_ratio_features import add_finance_ratios  # noqa: E402
from survey_features import add_survey_features  # noqa: E402
from text_features import TEXT_COL, build_model, nested_text_pred  # noqa: E402
from embedding_features import (  # noqa: E402
    EMB_LR_PARAMS, ORG_EMB_C, ORG_EMB_DIM, build_embed_model, load_embeddings,
)
from expert_groups import (  # noqa: E402
    e0_anchor_cols, e1_finance_cols, e2_survey_cols, e6_manual_cols,
)
from meta_blend import (  # noqa: E402
    apply_meta, fit_meta, select_alpha, to_logit, uniform_blend,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
DX_SCORE_COL = "DX展望_購入確率"
SURVEY_LR = dict(C=1.0, max_iter=2000, random_state=0)
EXPERTS = ("E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual")


def build_features(train, test, finance_ratios=False, survey_step=False):
    """train/test に行単位の決定的特徴を積む。fold内スコアはここでは足さない。

    finance_ratios=True のとき H21 の財務比4本も積む。この列は E1 だけが使い、
    E0 は `e0_anchor_cols` で除外する。
    survey_step=True のとき H22 の閾値指標4本を積む。同様に E2 だけが使う。
    """
    tp = harness_preprocess(train)
    y = tp[TARGET].values
    X = add_h1(train, tp.drop(columns=[TARGET]))

    cat_categories = {c: tp[c].cat.categories for c in CAT_COLS}
    tp_te = sub_preprocess(test, cat_categories=cat_categories)
    Xte = add_h1(test, tp_te.drop(columns=[TARGET], errors="ignore"))

    adders = [add_org_manual_features, add_dx_outlook_manual_features,
              add_company_overview_six_rate_flags]
    if finance_ratios:
        adders.append(add_finance_ratios)
    if survey_step:
        adders.append(lambda raw, df: add_survey_features(raw, df,
                                                          groups=("step",)))
    for add in adders:
        X, Xte = add(train, X), add(test, Xte)
    return X, y, Xte[X.columns]


def e4_inputs(train, test, org_embed):
    """E4(組織図エキスパート)の入力とモデル生成関数を返す。

    org_embed=True で TF-IDF+LR → OpenAI embedding+LR に **差し替える**（追加ではない）。
    入力を「文字列の1次元配列」から「埋め込みの2次元行列」に替えるだけで、
    fold スライス `e4[tr]` も fit も predict もそのまま動く。

    **E0 は触らない**。E0 が内部で持つ `組織図_購入確率` は現行の TF-IDF のまま
    にしてあるので、E0 の予測は org_embed の有無で完全一致する。これにより
    ホールドアウト検証で rep 単位の厳密なペアが取れる（`--survey-step` と同じ設計）。
    """
    if org_embed:
        tr, te = load_embeddings("org", dim=ORG_EMB_DIM)
        params = {**EMB_LR_PARAMS, "C": ORG_EMB_C}
        return tr, te, lambda: build_embed_model(params)
    return load_org_text(train), load_org_text(test), build_org_model


def _lgbm(Xtr, ytr, Xva, yva, Xte, seed):
    m = lgb.train({**BASE_PARAMS, "seed": seed}, lgb.Dataset(Xtr, ytr),
                  num_boost_round=2000, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n = m.best_iteration
    return m.predict(Xva, num_iteration=n), m.predict(Xte, num_iteration=n)


def _survey_lr(Xtr, ytr, Xva, Xte):
    model = make_pipeline(StandardScaler(), LogisticRegression(**SURVEY_LR))
    model.fit(Xtr.fillna(0), ytr)
    return (model.predict_proba(Xva.fillna(0))[:, 1],
            model.predict_proba(Xte.fillna(0))[:, 1])


def compute_expert_preds(seed, cache=True, finance_ratios=False,
                         survey_step=False, org_embed=False):
    """各エキスパートの OOF と test 予測を返す。重いのでnpzにキャッシュする。

    キャッシュ名に構成タグを含める。専用列のあり/なしで中身が変わるので、
    タグを分けないと片方の結果をもう片方に使い回してしまう。
    """
    tag = (("_ratio" if finance_ratios else "") + ("_step" if survey_step else "")
           + ("_orgemb" if org_embed else ""))
    path = os.path.join(OUT_DIR, f"_experts_seed{seed}{tag}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        print(f"(キャッシュを利用: {path})")
        return d["y"], {n: d[f"oof_{n}"] for n in EXPERTS}, \
            {n: d[f"te_{n}"] for n in EXPERTS}

    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, finance_ratios=finance_ratios,
                               survey_step=survey_step)
    txt, txt_te = (train[TEXT_COL].fillna("").astype(str).values,
                   test[TEXT_COL].fillna("").astype(str).values)
    org, org_te = load_org_text(train), load_org_text(test)  # E0 が使う（常にTF-IDF）
    e4, e4_te, e4_model = e4_inputs(train, test, org_embed)
    c0 = e0_anchor_cols(X.columns)
    c1, c2, c6 = (e1_finance_cols(with_ratios=finance_ratios),
                  e2_survey_cols(with_step=survey_step),
                  e6_manual_cols(X.columns))

    oof = {n: np.zeros(len(X)) for n in EXPERTS}
    te = {n: [] for n in EXPERTS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"  fold {k}/5 ...", flush=True)
        Xtr, Xva = X.iloc[tr], X.iloc[va]

        m_txt = build_model().fit(txt[tr], y[tr])
        oof["E3_dx_text"][va] = m_txt.predict_proba(txt[va])[:, 1]
        te["E3_dx_text"].append(m_txt.predict_proba(txt_te)[:, 1])

        m_org = e4_model().fit(e4[tr], y[tr])
        oof["E4_org_text"][va] = m_org.predict_proba(e4[va])[:, 1]
        te["E4_org_text"].append(m_org.predict_proba(e4_te)[:, 1])

        # E0: 現行と同じくテキストスコアをネストCVで作って94列にする
        t_tr, t_va, t_te = _nested_text(txt[tr], y[tr], txt[va], txt_te)
        o_tr, o_va, o_te = fold_org_preds(org[tr], y[tr], org[va], org_te)
        A_tr, A_va, A_te = Xtr[c0].copy(), Xva[c0].copy(), Xte[c0].copy()
        for col, v in ((DX_SCORE_COL, (t_tr, t_va, t_te)),
                       (ORG_SCORE_COL, (o_tr, o_va, o_te))):
            A_tr[col], A_va[col], A_te[col] = v
        p, q = _lgbm(A_tr, y[tr], A_va, y[va], A_te, seed)
        oof["E0_anchor"][va] = p
        te["E0_anchor"].append(q)

        for name, cols in (("E1_finance", c1), ("E6_manual", c6)):
            p, q = _lgbm(Xtr[cols], y[tr], Xva[cols], y[va], Xte[cols], seed)
            oof[name][va] = p
            te[name].append(q)

        p, q = _survey_lr(Xtr[c2], y[tr], Xva[c2], Xte[c2])
        oof["E2_survey"][va] = p
        te["E2_survey"].append(q)

    te = {n: np.mean(v, axis=0) for n, v in te.items()}
    np.savez(path, y=y, **{f"oof_{n}": oof[n] for n in EXPERTS},
             **{f"te_{n}": te[n] for n in EXPERTS})
    print(f"保存: {path}")
    return y, oof, te


def _nested_text(txt_tr, y_tr, txt_va, txt_te):
    """E0用。nested_text_pred に test 予測を足した版。"""
    o, v = nested_text_pred(txt_tr, y_tr, txt_va)
    full = build_model().fit(txt_tr, y_tr)
    return o, v, full.predict_proba(txt_te)[:, 1]


def _scores(y, p):
    f1s = [f1_score(y, (p >= t).astype(int)) for t in THS]
    b = int(np.argmax(f1s))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1s[b], th=THS[b])


def blend_oof(y, oof, te, seed, alpha=None, experts=EXPERTS, target="zero"):
    """メタ学習器を外側fold内でfitして、合成OOFとtest確率を返す。

    alpha=None なら **外側trainの中だけで** alpha を選ぶ。foldごとに選び直すので、
    評価対象の外側validはalpha選択にも重み推定にも一切寄与しない。

    experts で構成員を絞れる。キャッシュには常に全エキスパートの予測が入っているので、
    部分集合の評価に再計算は要らない。
    """
    Z = to_logit(np.column_stack([oof[n] for n in experts]))
    Zte = to_logit(np.column_stack([te[n] for n in experts]))
    out = np.zeros(len(y))
    weights, te_parts, alphas = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Z, y):
        a = (alpha if alpha is not None
             else select_alpha(Z[tr], y[tr], seed=seed, target=target)[0])
        alphas.append(a)
        w, b = fit_meta(Z[tr], y[tr], alpha=a, target=target)
        out[va] = apply_meta(Z[va], w, b)
        weights.append(w)
        te_parts.append(apply_meta(Zte, w, b))
    return out, np.mean(te_parts, axis=0), np.mean(weights, axis=0), alphas


def _resolve_threshold(blend, blend_te, y, oof_th, test_pos=None, threshold=None):
    """提出に使う閾値を決める。既定は OOF最適(oof_th)。

    test_pos で test の予測正例数を、threshold で閾値そのものを直接指定できる。
    ホールドアウト5repでは oof_best が fold_mean / rate / half_f1 のいずれにも
    勝っている（現本命で 1/5, 1/5, 0/5）ので、既定から外すのは
    「Public寄りに振るヘッジ」という明示的な意思決定のときだけにすること。
    """
    if threshold is not None:
        th, why = float(threshold), "手動指定"
    elif test_pos is not None:
        th = float(np.quantile(blend_te, 1 - test_pos / len(blend_te)))
        why = f"test正例{test_pos}本に合わせたrate指定"
    else:
        return float(oof_th), "OOF最適(oof_best)"
    d = f1_score(y, (blend >= th).astype(int)) - f1_score(y, (blend >= oof_th).astype(int))
    print(f"閾値 {th:.4f} ({why})  OOF最適 {oof_th:.3f} からの ΔOOF F1 {d:+.4f}")
    return th, why


def run(seed, alpha, submit, no_cache, experts=EXPERTS, finance_ratios=False,
        test_pos=None, threshold=None, target="zero", survey_step=False,
        org_embed=False):
    y, oof, te = compute_expert_preds(seed, cache=not no_cache,
                                      finance_ratios=finance_ratios,
                                      survey_step=survey_step,
                                      org_embed=org_embed)
    if target != "zero":
        print(f"メタ重みの縮小先: {target}")
    if finance_ratios:
        print("E1に H21 財務比4本を追加した構成")
    if survey_step:
        print("E2に H22 閾値指標4本を追加した構成")
    if org_embed:
        print(f"E4 を組織図 embedding+LR に差し替えた構成 "
              f"(dim={ORG_EMB_DIM}, C={ORG_EMB_C})")
    if tuple(experts) != EXPERTS:
        print(f"構成員: {list(experts)}")

    rows = {n: _scores(y, oof[n]) for n in experts}
    Z = to_logit(np.column_stack([oof[n] for n in experts]))
    rows["-- uniform blend"] = _scores(y, uniform_blend(Z))
    blend, blend_te, w, alphas = blend_oof(y, oof, te, seed, alpha,
                                           experts=experts, target=target)
    key = "-- meta (alpha=auto)" if alpha is None else f"-- meta (alpha={alpha})"
    rows[key] = _scores(y, blend)

    table = pd.DataFrame(rows).T
    print(f"\n=== OOFスコア (seed={seed}) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if alpha is None:
        print(f"外側foldごとに選ばれたalpha: {alphas}")

    print("\n=== 学習された重み (5fold平均, 非負) ===")
    for n, v in sorted(zip(experts, w), key=lambda x: -x[1]):
        bar = "#" * int(round(v * 40))
        print(f"  {n:12s} {v:6.3f} {bar}")

    a = table.loc["E0_anchor"]
    m = table.loc[key]
    print(f"\nE0単体 -> メタ:  AUC {m.auc - a.auc:+.4f} / AP {m.ap - a.ap:+.4f}"
          f" / F1 {m.f1 - a.f1:+.4f}")

    if submit:
        th, _ = _resolve_threshold(blend, blend_te, y, m.th, test_pos, threshold)
        label = (blend_te >= th).astype(int)
        test = pd.read_csv("data/test.csv")
        sample = pd.read_csv("data/sample_submit.csv", header=None,
                             names=["企業ID", "購入フラグ"])
        pred = pd.DataFrame({"企業ID": test["企業ID"].values, "pred": label})
        sub = sample[["企業ID"]].merge(pred, on="企業ID", how="left")
        assert sub["pred"].notna().all(), "予測欠損あり"
        tag = "auto" if alpha is None else str(alpha)
        n_tag = "" if tuple(experts) == EXPERTS else f"_{len(experts)}experts"
        r_tag = (("_ratio" if finance_ratios else "")
                 + ("_step" if survey_step else "")
                 + ("_orgemb" if org_embed else ""))
        t_tag = (f"_pos{test_pos}" if test_pos is not None
                 else f"_th{threshold}" if threshold is not None else "")
        out = (f"submission/submission_ensemble_experts_seed{seed}"
               f"_a{tag}{n_tag}{r_tag}{t_tag}.csv")
        sub.assign(pred=sub["pred"].astype(int)).to_csv(
            out, index=False, header=False, lineterminator="\n")
        print(f"保存: {out} 正例={int(label.sum())} (th={th:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=None,
                   help="メタのL2強度。省略時は外側trainの中だけで自動選択")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--drop", nargs="*", default=[], choices=list(EXPERTS),
                   help="構成員から外すエキスパート（キャッシュは全員分あるので再計算不要）")
    p.add_argument("--finance-ratios", action="store_true",
                   help="E1 に H21 財務比4本を足す（E0は現行94列のまま）")
    p.add_argument("--survey-step", action="store_true",
                   help="E2 に H22 閾値指標4本を足す（E0は現行94列のまま）")
    p.add_argument("--org-embed", action="store_true",
                   help="E4 を組織図の OpenAI embedding+LR に差し替える（E0は不変）")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--test-pos", type=int,
                   help="testの予測正例数を指定して閾値を逆算（既定はOOF最適）")
    g.add_argument("--threshold", type=float, help="閾値を直接指定")
    p.add_argument("--meta-target", default="zero", choices=["zero", "uniform"],
                   help="メタ重みの縮小先。uniform は等分ブレンドへ縮む")
    a = p.parse_args()
    run(a.seed, a.alpha, a.submit, a.no_cache,
        experts=tuple(n for n in EXPERTS if n not in a.drop),
        finance_ratios=a.finance_ratios,
        test_pos=a.test_pos, threshold=a.threshold, target=a.meta_target,
        survey_step=a.survey_step, org_embed=a.org_embed)
