"""H92: what do they plan to buy, and is it *this* product?

The product is a DX training material for employees -- an e-learning platform
plus a skill assessment tool (signate_techocean_student_cup_2026.md).  Every
text feature built so far reads *intensity* ("how eager are they about DX /
about education"): H14's education counts, H37 governance/priority, the three
LLM axes, H67-H69's template handling.  **Nothing reads modality or scope** --
whether the education they describe is the kind this product sells.

The hypothesis under test: rows the model calls buyers but that do not buy
(FP) are companies whose plans are out of this product's scope -- workshops,
OJT, qualification support, in-house academies -- so a training-material
vendor never closes them.

Modes:

``scope``
    Deterministic keyword features over the *education sentences only*, split
    into in-scope (e-learning / assessment / company-wide literacy / bought
    from outside / newly introduced) and out-of-scope (classroom / hands-on /
    OJT / qualifications / dispatch / in-house build / already owned).
    Reports univariate AUC plus the error structure the hypothesis predicts:
    FP vs TP and FN vs TN.

``ceiling``
    Model-free bound on the whole idea.  The DX text embedding is projected
    onto PC1 (the DX-maturity axis) and the remainder is measured on its own,
    inside the p>0.5 group where the FPs live, and as a ninth expert in the
    production blend.  If the text carries scope information that matters,
    it has to be in that remainder.

``angles``
    The same hypothesis through four instruments that share no assumptions
    with the keyword dictionary, because a dictionary I wrote by hand is one
    instrument and its null could be my vocabulary's fault:

      A 前提の検査   which expert actually pushed each FP up.  If the text
                     expert is not what makes them look like buyers, the
                     premise "the text says they want to buy" is wrong.
      B 語の自動選択 char n-gram TF-IDF over the education sentences,
                     supervised on FP vs TP, with a permutation null.  The
                     data picks the words; then read the coefficients.
      C 教師なし話題 NMF topics over train+test education sentences, no
                     labels.  Do the "what they will do" clusters that emerge
                     line up with buying, and are out-of-scope topics really
                     the FP ones?
      D 意味的距離   cosine between each company's education sentences and a
                     written description of *this product*, against five
                     out-of-scope descriptions, with a local e5 encoder.
                     The instrument is validated first: it must recover the
                     keyword flags before its null means anything.

Run:
  .venv/bin/python exp/compare_material_scope.py --mode scope
  .venv/bin/python exp/compare_material_scope.py --mode ceiling
  .venv/bin/python exp/compare_material_scope.py --mode angles
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_short_term_debt_gate import _blend_oof_p  # noqa: E402
from decision import verdict  # noqa: E402
from diagnose_meta_alpha import CONFIGS, _load  # noqa: E402
from ensemble_experts import THS, blend_oof  # noqa: E402

EXPERTS = ["E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual", "E7_cross", "E0b_linear"]
RESID_EXPERT = "E8_text_resid"
EMB = "data/_emb_dx_outlook_text-embedding-3-large.npz"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42)
TEXT_COL = "今後のDX展望"

EDU_SENTENCE = re.compile(
    "教育|研修|リスキリング|リスキル|人材育成|人財育成|人材開発|ラーニング|学習"
    "|スキル|リテラシー|講座|セミナー|勉強会|ワークショップ|OJT|資格|育成")

# 商材＝従業員向けDX教育のe-learning + スキルアセスメント。
IN_SCOPE = {
    "eラーニング系": "eラーニング|Eラーニング|イーラーニング|e-?learning|オンライン学習"
                     "|オンラインラーニング|オンライン研修|LMS|学習プラットフォーム"
                     "|学習基盤|動画教材|デジタル教材|オンデマンド|自己学習|マイクロラーニング",
    "アセスメント系": "アセスメント|スキル可視化|スキルマップ|スキル診断|習熟度|到達度"
                      "|レベル判定|スキル測定|認定制度|検定",
    "全社リテラシー": "全社員|全従業員|全社的な?(教育|研修|リテラシー)"
                      "|リテラシー(向上|底上げ|教育)|底上げ",
    "外部調達": "外部(の)?(専門家|講師|ベンダー|パートナー|サービス)|ベンダー|委託"
                "|外部研修|商材|サービスを活用",
    "新規導入": "新設|新たに(導入|整備|構築)|新規に|刷新|導入します|導入する|拡充します",
}
OUT_SCOPE = {
    "集合・現場型": "集合研修|ワークショップ|ハンズオン|演習|勉強会|合宿|OJT|現場研修"
                    "|実践型|PBL|実地|対面",
    "資格支援": "資格取得|資格支援|資格取得支援|受験費用",
    "社外派遣": "派遣|ハッカソン|カンファレンス|大学(と|との)連携|社外コミュニティ|留学"
                "|イベントへの参加",
    "内製化": "内製|社内講師|自前|社内大学|アカデミー|自社で(開発|作成)",
    "既に保有": "既存の|導入済み|確立済み|これまで(導入|実施)|従来の(研修|教育|プログラム)"
                "|一定の成果",
}


def education_context(text):
    """教育に触れている文だけを残す。他のDX話題は落とす。"""
    return " ".join(s for s in re.split(r"(?<=。)", str(text))
                    if EDU_SENTENCE.search(s))


def build_material_scope_features(raw):
    """ラベル非依存。教育文に対する語彙フラグと、その差分スコア。"""
    edu = raw[TEXT_COL].fillna("").astype(str).map(education_context)
    out = {}
    for name, pattern in {**IN_SCOPE, **OUT_SCOPE}.items():
        out[name] = edu.str.contains(pattern, regex=True).astype(float).values
    out["スコープ適合スコア"] = (sum(out[k] for k in IN_SCOPE)
                                 - sum(out[k] for k in OUT_SCOPE))
    out["教育文の文字数比"] = (edu.str.len()
                               / raw[TEXT_COL].fillna("").astype(str).str.len()
                               .clip(lower=1)).values
    return out


def _auc_ci(v, y, mask, rng, n_boot=2000):
    v, yy = np.asarray(v, float)[mask], y[mask]
    if len(np.unique(yy)) < 2:
        return None
    a, n, bs = roc_auc_score(yy, v), int(mask.sum()), []
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True)
        if 0 < yy[i].sum() < n:
            bs.append(roc_auc_score(yy[i], v[i]))
    lo, hi = np.quantile(bs, [.025, .975])
    return a, lo, hi


def _rate_diff(v, a_mask, b_mask):
    a, b = v[a_mask].mean(), v[b_mask].mean()
    se = np.sqrt(a * (1 - a) / a_mask.sum() + b * (1 - b) / b_mask.sum())
    return a, b, a - b, 1.96 * se


def run_scope(seed, config):
    raw = pd.read_csv("data/train.csv")
    y, Z, _ = _load(CONFIGS[config], seed)
    p = _blend_oof_p(y, Z, seed, "auto")
    F = build_material_scope_features(raw)
    rng = np.random.default_rng(0)
    allm = np.ones(len(y), bool)
    band, hi, lo = (p > 0.1) & (p < 0.9), p > 0.5, p <= 0.5
    fp, tp, fn, tn = hi & (y == 0), hi & (y == 1), lo & (y == 1), lo & (y == 0)

    print(f"教育に触れる文が1つも無い行: "
          f"{int((pd.Series(raw[TEXT_COL].fillna('')).map(education_context).str.len() == 0).sum())}"
          f" / {len(raw)}\n")
    print(f"{'特徴':<16}{'出現率':>7}{'全体AUC':>9}{'  95%CI':>17}"
          f"{'p>0.5内':>9}{'  95%CI':>17}{'帯内':>7}")
    for k in [*IN_SCOPE, *OUT_SCOPE, "スコープ適合スコア"]:
        v = F[k]
        r1, r2, r3 = (_auc_ci(v, y, allm, rng), _auc_ci(v, y, hi, rng),
                      _auc_ci(v, y, band, rng))
        mark = "  ← 全体で有意" if (r1[1] > 0.5 or r1[2] < 0.5) else ""
        print(f"{k:<16}{v.mean():>7.3f}{r1[0]:>9.3f}  [{r1[1]:.3f},{r1[2]:.3f}]"
              f"{r2[0]:>9.3f}  [{r2[1]:.3f},{r2[2]:.3f}]{r3[0]:>7.3f}{mark}")

    print(f"\n=== 仮説の直接検定: FP({int(fp.sum())}件) vs TP({int(tp.sum())}件) ===")
    print("  仮説が正しければ in-scope は FP<TP、out-of-scope は FP>TP になるはず\n")
    print(f"{'特徴':<16}{'FP':>7}{'TP':>7}{'差':>8}{'  95%CI':>18}{'  向き':>8}")
    for k in [*IN_SCOPE, *OUT_SCOPE]:
        a, b, d, h = _rate_diff(F[k], fp, tp)
        want = "FP<TP" if k in IN_SCOPE else "FP>TP"
        got = "FP<TP" if d < 0 else "FP>TP"
        print(f"{k:<16}{a:>7.3f}{b:>7.3f}{d:>+8.3f}  [{d - h:+.3f},{d + h:+.3f}]"
              f"{'  一致' if want == got else '  逆':>8}")

    print(f"\n=== 裏側: FN({int(fn.sum())}件) vs TN({int(tn.sum())}件) ===")
    print(f"{'特徴':<16}{'FN':>7}{'TN':>7}{'差':>8}{'  95%CI':>18}")
    for k in [*IN_SCOPE, *OUT_SCOPE]:
        a, b, d, h = _rate_diff(F[k], fn, tn)
        mark = "  ← 0を跨がない" if abs(d) > h else ""
        print(f"{k:<16}{a:>7.3f}{b:>7.3f}{d:>+8.3f}  [{d - h:+.3f},{d + h:+.3f}]{mark}")


def _text_parts():
    z = np.load(EMB, allow_pickle=True)
    E = np.vstack([z["train"], z["test"]]).astype(np.float64)
    ntr = len(z["train"])
    pca = PCA(n_components=1, random_state=0).fit(E)   # ラベル不使用
    resid = E - pca.inverse_transform(pca.transform(E))
    return E[:ntr], E[ntr:], resid[:ntr], resid[ntr:], pca.transform(E)[:ntr]


def _cv_oof(X, y, seeds=(0, 1, 2), C=0.03):
    out = []
    for s in seeds:
        o = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True,
                                      random_state=s).split(X, y):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=4000))
            o[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        out.append(o)
    return np.mean(out, axis=0)


def run_ceiling(seed, config, n_perm):
    Etr, Ete, Rtr, Rte, pc1 = _text_parts()
    y, Z, _ = _load(CONFIGS[config], seed)
    p = _blend_oof_p(y, Z, seed, "auto")
    band, hi = (p > 0.1) & (p < 0.9), p > 0.5

    print("=== 1. DX展望テキストは何次元ぶんの情報か ===")
    for name, X in (("全3072次元", Etr), ("PC1だけ", pc1),
                    ("PC1を抜いた残り", Rtr)):
        o = _cv_oof(X, y)
        print(f"  {name:<16} 全体AUC {roc_auc_score(y, o):.3f}"
              f"   帯内 {roc_auc_score(y[band], o[band]):.3f}"
              f"   p>0.5内 {roc_auc_score(y[hi], o[hi]):.3f}")

    print(f"\n=== 2. テキストだけで FP と TP を見分けられるか（{int(hi.sum())}行）===")
    sub = np.where(hi)[0]
    ysub = y[sub]
    obs = roc_auc_score(ysub, _cv_oof(Rtr[sub], ysub))
    rng = np.random.default_rng(0)
    null = np.array([roc_auc_score(yp := rng.permutation(ysub),
                                   _cv_oof(Rtr[sub], yp))
                     for _ in range(n_perm)])
    print(f"  観測AUC {obs:.3f}（PC1を抜いた残差）")
    print(f"  ラベル入替{n_perm}回の帰無: 平均 {null.mean():.3f} sd {null.std():.3f}"
          f"  95%点 {np.quantile(null, 0.95):.3f}")
    print(f"  経験的p値 {(null >= obs).mean():.3f}"
          f"  ({int((null >= obs).sum())}/{n_perm})")

    print("\n=== 3. 9本目のExpertとして本番の合成に足す（11seed screen）===")
    rows = []
    for s in SEEDS:
        d = np.load(CONFIGS[config].format(seed=s), allow_pickle=True)
        ys = d["y"]
        oof = {n: d["oof_" + n] for n in EXPERTS}
        te = {n: d["te_" + n] for n in EXPERTS}
        folds = list(StratifiedKFold(5, shuffle=True,
                                     random_state=s).split(Rtr, ys))
        r_oof = np.zeros(len(ys))
        for tr, va in folds:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.03, max_iter=4000))
            r_oof[va] = m.fit(Rtr[tr], ys[tr]).predict_proba(Rtr[va])[:, 1]
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.03, max_iter=4000)).fit(Rtr, ys)
        r_te = m.predict_proba(Rte)[:, 1]
        base, _, _, _ = blend_oof(ys, oof, te, seed=s, experts=EXPERTS)
        cand, _, _, _ = blend_oof(ys, {**oof, RESID_EXPERT: r_oof},
                                  {**te, RESID_EXPERT: r_te}, seed=s,
                                  experts=EXPERTS + [RESID_EXPERT])
        for arm, q in (("base", base), ("cand", cand)):
            rows.append({"seed": s, "arm": arm, "auc": roc_auc_score(ys, q),
                         "ap": average_precision_score(ys, q),
                         "f1": max(f1_score(ys, (q >= t).astype(int))
                                   for t in THS),
                         "solo": roc_auc_score(ys, r_oof)})
    r = pd.DataFrame(rows)
    b = r[r.arm == "base"].sort_values("seed").reset_index(drop=True)
    c = r[r.arm == "cand"].sort_values("seed").reset_index(drop=True)
    print(f"  残差テキスト単体AUC {r.solo.mean():.4f}")
    print(f"  base  auc {b.auc.mean():.5f}  ap {b.ap.mean():.5f}  f1 {b.f1.mean():.5f}")
    print(f"  cand  auc {c.auc.mean():.5f}  ap {c.ap.mean():.5f}  f1 {c.f1.mean():.5f}")
    line = []
    for metric in ("auc", "ap", "f1"):
        dd = c[metric].values - b[metric].values
        line.append(f"{metric} {dd.mean():+.5f} ({int((dd > 0).sum())}/{len(dd)})")
    v, _ = verdict(b, c)
    print("  Δ " + "  ".join(line) + f"   {v}")
    r.to_csv("exp/material_scope_ceiling.csv", index=False)
    print("\nsaved: exp/material_scope_ceiling.csv")


ANCHORS = {
    "商材そのもの": "従業員向けのDX教育のeラーニングプラットフォームを導入し、"
                    "デジタルスキルを測定・可視化するアセスメントツールを全社に展開する",
    "外:集合研修OJT": "集合研修やハンズオン演習、現場でのOJTを通じて社員を育成する",
    "外:資格支援": "資格取得を支援し、受験費用の補助や報奨金の制度を設ける",
    "外:内製": "社内に教育専門の部署や社内大学を作り、自前で研修プログラムを内製する",
    "外:社外派遣": "外部セミナーやカンファレンス、ハッカソンに社員を派遣して学ばせる",
    "外:教育以外DX": "工場設備のIoT化やデータ基盤の整備、基幹システムの刷新に投資する",
}
# アンカーが本当に「やること」を見ているかの検証ペア（アンカー -> 対応するキーワード印）
ANCHOR_CHECK = (("商材そのもの", "eラーニング系"), ("外:集合研修OJT", "集合・現場型"),
                ("外:資格支援", "資格支援"), ("外:内製", "内製化"),
                ("外:社外派遣", "社外派遣"))
E5_MODEL = "intfloat/multilingual-e5-base"


def _cv_auc(X, yy, seeds=3, C=1.0):
    scores = []
    for s in range(seeds):
        o = np.zeros(len(yy))
        for tr, va in StratifiedKFold(5, shuffle=True,
                                      random_state=s).split(X, yy):
            m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], yy[tr])
            o[va] = m.predict_proba(X[va])[:, 1]
        scores.append(roc_auc_score(yy, o))
    return float(np.mean(scores))


def _perm_report(X, yy, obs, rng, n_perm, label):
    null = np.array([_cv_auc(X, rng.permutation(yy), seeds=2)
                     for _ in range(n_perm)])
    print(f"  {label} CV-AUC {obs:.3f}   帰無(入替{n_perm}回) 平均 {null.mean():.3f}"
          f" sd {null.std():.3f} 95%点 {np.quantile(null, .95):.3f}"
          f"   p={(null >= obs).mean():.3f}")


def _angle_premise(raw, y, p, cache):
    from scipy.special import logit as _lg
    hi, lo = p > 0.5, p <= 0.5
    fp, tp, fn, tn = hi & (y == 0), hi & (y == 1), lo & (y == 1), lo & (y == 0)
    print("=" * 92)
    print("角度A: 前提の検査 — FPは『テキストが買いたそう』な行なのか")
    print("=" * 92)
    print("各Expertのlogit平均（高い=買うと言っている）\n")
    print(f"{'Expert':<14}{'FP':>9}{'TP':>9}{'FN':>9}{'TN':>9}{'FP-TP':>9}")
    e3 = None
    for name in EXPERTS:
        v = _lg(np.clip(cache["oof_" + name], 1e-6, 1 - 1e-6))
        if name == "E3_dx_text":
            e3 = v
        print(f"{name:<14}{v[fp].mean():>9.2f}{v[tp].mean():>9.2f}"
              f"{v[fn].mean():>9.2f}{v[tn].mean():>9.2f}"
              f"{v[fp].mean() - v[tp].mean():>+9.2f}")
    print(f"\n  テキスト単体(E3)で上位半分に入る割合: "
          f"FP {(e3[fp] > np.median(e3)).mean() * 100:.0f}%  "
          f"TP {(e3[tp] > np.median(e3)).mean() * 100:.0f}%  "
          f"（全体では定義上50%)")


def _angle_lexical(edu_all, y, p, n_train, rng, n_perm):
    from sklearn.feature_extraction.text import TfidfVectorizer
    hi = p > 0.5
    sub = np.where(hi)[0]
    ysub = y[sub]
    print("\n" + "=" * 92)
    print("角度B: 語をデータに選ばせる — 教育文のTF-IDFでFP vs TPを分類し係数を読む")
    print("=" * 92)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=3, sublinear_tf=True)
    X = vec.fit_transform(edu_all)[:n_train][sub]
    _perm_report(X, ysub, _cv_auc(X, ysub), rng, n_perm, "FP/TP識別")
    coef = LogisticRegression(C=1.0, max_iter=3000).fit(X, ysub).coef_[0]
    names = np.array(vec.get_feature_names_out())
    for label, idx in (("TP側(買う)", np.argsort(coef)[::-1][:12]),
                       ("FP側(買わない)", np.argsort(coef)[:12])):
        got = " / ".join(f"「{s.strip()}」" for s in names[idx] if s.strip())
        print(f"  {label}に効く文字列: {got}")


def _angle_topics(edu_all, y, p, n_train, n_topics=7):
    from janome.tokenizer import Tokenizer
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    hi = p > 0.5
    fp, tp = hi & (y == 0), hi & (y == 1)
    jt = Tokenizer()

    def words(text):
        return [t.surface for t in jt.tokenize(str(text))
                if t.part_of_speech.split(",")[0] in ("名詞", "動詞", "形容詞")
                and len(t.surface) > 1]

    print("\n" + "=" * 92)
    print("角度C: 教師なし話題 — 教育文が何種類の『やること』に分かれるか(NMF)")
    print("=" * 92)
    vec = TfidfVectorizer(analyzer=words, min_df=8, max_df=0.6, sublinear_tf=True)
    X = vec.fit_transform(edu_all)
    nmf = NMF(n_components=n_topics, random_state=0, init="nndsvda",
              max_iter=600).fit(X)
    W = nmf.transform(X)[:n_train]
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    feat = np.array(vec.get_feature_names_out())
    print(f"{'topic':<6}{'上位語':<50}{'主話題の購入率':>12}{'FP':>8}{'TP':>8}{'差':>8}")
    for k in range(n_topics):
        top = " ".join(feat[np.argsort(nmf.components_[k])[::-1][:7]])
        dom = W.argmax(1) == k
        rate = y[dom].mean() if dom.sum() > 5 else np.nan
        print(f"T{k:<5}{top:<50}{rate:>12.3f}{W[fp, k].mean():>8.3f}"
              f"{W[tp, k].mean():>8.3f}{W[fp, k].mean() - W[tp, k].mean():>+8.3f}")
    sub = np.where(hi)[0]
    print(f"\n  話題分布{n_topics}次元だけで FP/TP を分ける CV-AUC "
          f"{_cv_auc(W[sub], y[sub]):.3f}")
    return W


def _angle_anchors(raw, y, p, rng, n_perm):
    print("\n" + "=" * 92)
    print("角度D: 意味的距離 — 商材の説明文そのものとの近さ（ローカルe5）")
    print("=" * 92)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(E5_MODEL, local_files_only=True)
        mdl = AutoModel.from_pretrained(E5_MODEL, local_files_only=True).eval()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  スキップ: {E5_MODEL} をローカルで読めない ({type(exc).__name__})")
        return
    torch.set_num_threads(8)

    @torch.no_grad()
    def encode(texts, prefix, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok([prefix + str(t) for t in texts[i:i + batch]],
                      padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
        return np.vstack(out)

    edu = raw[TEXT_COL].fillna("").astype(str).map(education_context).values
    S = encode(edu, "passage: ") @ encode(list(ANCHORS.values()), "query: ").T
    names = list(ANCHORS)
    F = build_material_scope_features(raw)

    print("  計器の検証（この類似度は本当に『やること』を見ているか）")
    for anchor, flag in ANCHOR_CHECK:
        auc = roc_auc_score(F[flag], S[:, names.index(anchor)])
        print(f"    『{anchor}』との近さで キーワード{flag} を当てる AUC {auc:.3f}")

    hi, band = p > 0.5, (p > 0.1) & (p < 0.9)
    fp, tp = hi & (y == 0), hi & (y == 1)
    fit = S[:, 0] - S[:, 1:].max(1)
    allm = np.ones(len(y), bool)
    print(f"\n{'アンカー':<16}{'全体AUC':>9}{'  95%CI':>17}{'p>0.5内':>9}"
          f"{'  95%CI':>17}{'FP-TP':>9}")
    for i, name in enumerate([*names, "スコープ適合(意味)"]):
        v = fit if i == len(names) else S[:, i]
        r1, r2 = _auc_ci(v, y, allm, rng), _auc_ci(v, y, hi, rng)
        print(f"{name:<16}{r1[0]:>9.3f}  [{r1[1]:.3f},{r1[2]:.3f}]"
              f"{r2[0]:>9.3f}  [{r2[1]:.3f},{r2[2]:.3f}]"
              f"{v[fp].mean() - v[tp].mean():>+9.4f}")
    r3 = _auc_ci(fit, y, band, rng)
    print(f"  スコープ適合(意味) の帯内AUC {r3[0]:.3f} [{r3[1]:.3f}, {r3[2]:.3f}]")
    sub = np.where(hi)[0]
    _perm_report(S[sub], y[sub], _cv_auc(S[sub], y[sub]), rng, n_perm,
                 "6アンカーの類似度ベクトルで FP/TP")


def run_angles(seed, config, n_perm):
    raw = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y, Z, _ = _load(CONFIGS[config], seed)
    p = _blend_oof_p(y, Z, seed, "auto")
    cache = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
    edu_all = pd.concat([raw[TEXT_COL].fillna(""),
                         test[TEXT_COL].fillna("")]).map(education_context).values
    rng = np.random.default_rng(0)
    _angle_premise(raw, y, p, cache)
    _angle_lexical(edu_all, y, p, len(raw), rng, n_perm)
    _angle_topics(edu_all, y, p, len(raw))
    _angle_anchors(raw, y, p, rng, n_perm)


def _crossfit_head(X_tr, X_te, y, folds, C=0.3):
    oof = np.zeros(len(y))
    for tr, va in folds:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=4000)).fit(X_tr[tr], y[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(C=C, max_iter=4000)).fit(X_tr, y)
    return oof, m.predict_proba(X_te)[:, 1]


def run_candidates(seeds, config):
    """4節までで作った特徴を、実際に9本目として判定装置にかける。

    単体AUCが基準にならないことを見るための節。合成pとの相関と残差AUCも出す。
    """
    raw = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    keys = [*IN_SCOPE, *OUT_SCOPE]
    f_tr, f_te = build_material_scope_features(raw), build_material_scope_features(test)
    flags_tr = np.column_stack([f_tr[k] for k in keys])
    flags_te = np.column_stack([f_te[k] for k in keys])
    cands = {
        "scope_10flags": (flags_tr, flags_te),
        "既に保有のみ": (f_tr["既に保有"].reshape(-1, 1), f_te["既に保有"].reshape(-1, 1)),
        "スコープ適合スコアのみ": (f_tr["スコープ適合スコア"].reshape(-1, 1),
                                  f_te["スコープ適合スコア"].reshape(-1, 1)),
    }
    rows = []
    for seed in seeds:
        cache = np.load(CONFIGS[config].format(seed=seed), allow_pickle=True)
        y = cache["y"]
        oof = {n: cache["oof_" + n] for n in EXPERTS}
        te = {n: cache["te_" + n] for n in EXPERTS}
        folds = list(StratifiedKFold(5, shuffle=True,
                                     random_state=seed).split(flags_tr, y))
        base, _, _, _ = blend_oof(y, oof, te, seed=seed, experts=EXPERTS)
        base_logit = np.log(np.clip(base, 1e-6, 1 - 1e-6)
                            / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
        rows.append({"seed": seed, "arm": "base", **_metrics(y, base)})
        for name, (A, B) in cands.items():
            o, t = _crossfit_head(A, B, y, folds)
            cand, _, _, _ = blend_oof(y, {**oof, "E8": o}, {**te, "E8": t},
                                      seed=seed, experts=EXPERTS + ["E8"])
            lo = np.log(np.clip(o, 1e-6, 1 - 1e-6) / (1 - np.clip(o, 1e-6, 1 - 1e-6)))
            fitted = LinearRegression().fit(base_logit.reshape(-1, 1),
                                            lo).predict(base_logit.reshape(-1, 1))
            rows.append({"seed": seed, "arm": name, "solo": roc_auc_score(y, o),
                         "resid": roc_auc_score(y, lo - fitted),
                         "corr": spearmanr(lo, base_logit).statistic,
                         **_metrics(y, cand)})
    r = pd.DataFrame(rows)
    b = r[r.arm == "base"].sort_values("seed").reset_index(drop=True)
    print("=== 実物を判定装置にかける（本番外側fold）===\n")
    print(f"{'候補':<22}{'単体AUC':>9}{'合成と相関':>10}{'残差AUC':>9}"
          f"{'ΔAUC':>10}{'ΔAP':>9}{'ΔAP>0':>8}  判定")
    for name in cands:
        c = r[r.arm == name].sort_values("seed").reset_index(drop=True)
        v, _ = verdict(b, c)
        print(f"{name:<22}{c.solo.mean():>9.3f}{c['corr'].mean():>10.3f}"
              f"{c.resid.mean():>9.3f}{(c.auc - b.auc).mean():>+10.4f}"
              f"{(c.ap - b.ap).mean():>+9.4f}{int((c.ap > b.ap).sum()):>6}/{len(c)}  {v}")
    print("\n  残差AUC 0.55 が損益分岐（§9.4）。単体AUCで足切りしないこと。")
    r.to_csv("exp/material_scope_candidates.csv", index=False)
    print("saved: exp/material_scope_candidates.csv")


def _metrics(y, p):
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
            "f1": max(f1_score(y, (p >= t).astype(int)) for t in THS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="scope",
                    choices=("scope", "ceiling", "angles", "candidates"))
    ap.add_argument("--config", default="h2h7both4", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=60)
    args = ap.parse_args()
    if args.mode == "scope":
        run_scope(args.seed, args.config)
    elif args.mode == "angles":
        run_angles(args.seed, args.config, args.n_perm)
    elif args.mode == "candidates":
        run_candidates(SEEDS, args.config)
    else:
        run_ceiling(args.seed, args.config, args.n_perm)


if __name__ == "__main__":
    main()
