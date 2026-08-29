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
from dx_outlook_features import (  # noqa: E402
    add_dx_outlook_manual_features, add_dx_outlook_priority_features,
)
from company_overview_features import add_company_overview_six_rate_flags  # noqa: E402
from cross_features import add_cross_features  # noqa: E402
from finance_ratio_features import FINANCE_RATIO_COLS, add_finance_ratios  # noqa: E402
from survey_features import add_survey_features, add_h57_subscales  # noqa: E402
from text_features import TEXT_COL, build_model, nested_text_pred  # noqa: E402
from embedding_features import (  # noqa: E402
    CONCAT_EMB_C, CONCAT_EMB_DIM, CONCAT_SLUGS, EMB_LR_PARAMS, ORG_EMB_C,
    ORG_EMB_DIM, H30_ABSDIFF_C, H30_ABSDIFF_DIM, build_embed_model,
    load_absdiff_embeddings, load_concat_embeddings, load_embeddings,
)
from expert_groups import (  # noqa: E402
    FINANCE_COLS, FINANCE_ISNA_COLS, SCALE_COLS, SURVEY_ISNA_COLS,
    e0_anchor_cols, e1_finance_cols, e2_survey_cols, e6_manual_cols,
    e7_cross_cols, route_h2_anchor_cols,
)
from meta_blend import (  # noqa: E402
    apply_meta, fit_meta, select_alpha, select_alpha_1se, to_logit,
    uniform_blend,
)
from gam_model import GAM_C, fit_gam  # noqa: E402
from embedding_knn import (  # noqa: E402
    KNN_DIM, KNN_DISTANCE_SCALE, KNN_NEIGHBORS, KNN_PRIOR_STRENGTH,
    KNN_SLUGS, build_knn_model, load_knn_embeddings,
)
from llm_make_or_buy_features import (  # noqa: E402
    LLM_FEATURE_COLS, add_llm_make_or_buy_features,
)
from hypotheses import h2_financial_health, h7_cash_slack  # noqa: E402
from readiness_to_scale_features import READINESS_COLS, add_readiness_features  # noqa: E402

H2_RESTORE_COLS = ("自己資本比率", "ROA")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
THS = np.arange(0.05, 0.95, 0.005)
DX_SCORE_COL = "DX展望_購入確率"
SURVEY_LR = dict(C=1.0, max_iter=2000, random_state=0)
CROSS_LR = dict(C=1.0, max_iter=2000, random_state=0)
# H28 E0b: 田中さん nb018 の設定（one-hot + 標準化 + C=0.1）を出発点にする。
LINEAR_C = 0.1
LINEAR_LR = dict(max_iter=5000, random_state=0)
EXPERTS = ("E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
           "E4_org_text", "E6_manual")
E7_NAME = "E7_cross"
E0B_NAME = "E0b_linear"
E0B_LOG_NAME = "E0b_signed_log"
E8_NAME = "E8_text_absdiff"
GAM_NAME = "E0c_gam"
E9_NAME = "E9_embed_knn"


def expert_names(cross=False, linear=False, h30_absdiff=False, gam=False,
                 embed_knn=False, linear_log_extra=False):
    """構成員の名前。既存6本は常に不変で、後ろに足していく。

    cross=True  … H27 E7（不満×財務の交互作用12列）
    linear=True … H28 E0b（E0と同じ94列に one-hot + LR）
    linear_log_extra=True … raw E0bに符号付き対数版E0bを追加
    """
    names = EXPERTS
    if cross:
        names += (E7_NAME,)
    if linear:
        names += (E0B_NAME,)
    if linear_log_extra:
        names += (E0B_LOG_NAME,)
    if h30_absdiff:
        names += (E8_NAME,)
    if gam:
        names += (GAM_NAME,)
    if embed_knn:
        names += (E9_NAME,)
    return names


def build_features(train, test, finance_ratios=False, survey_step=False,
                   cross=False, h37_priority=False, llm_makebuy=False,
                   h2_restore=False, h21_restore=False, h49_restore=False,
                   h57_subscales=False, h7_restore=False, h58=False):
    """train/test に行単位の決定的特徴を積む。fold内スコアはここでは足さない。

    finance_ratios=True のとき H21 の財務比4本も積む。この列は E1 だけが使い、
    E0 は `e0_anchor_cols` で除外する。
    survey_step=True のとき H22 の閾値指標4本を積む。同様に E2 だけが使う。
    cross=True のとき H27 の不満×財務12列を積む。同様に E7 だけが使う。
    """
    tp = harness_preprocess(train)
    y = tp[TARGET].values
    X = add_h1(train, tp.drop(columns=[TARGET]))

    cat_categories = {c: tp[c].cat.categories for c in CAT_COLS}
    tp_te = sub_preprocess(test, cat_categories=cat_categories)
    Xte = add_h1(test, tp_te.drop(columns=[TARGET], errors="ignore"))

    adders = [add_org_manual_features, add_dx_outlook_manual_features,
              add_company_overview_six_rate_flags]
    if h37_priority:
        adders.append(add_dx_outlook_priority_features)
    if finance_ratios or h21_restore:
        adders.append(lambda raw, df: add_finance_ratios(raw, df, h58=h58))
    if survey_step:
        adders.append(lambda raw, df: add_survey_features(raw, df,
                                                          groups=("step",)))
    if cross:
        adders.append(add_cross_features)
    if h49_restore:
        adders.append(add_readiness_features)
    if h2_restore:
        adders.append(h2_financial_health)
    if h7_restore:
        adders.append(h7_cash_slack)
    if h57_subscales:
        adders.append(add_h57_subscales)
    for add in adders:
        X, Xte = add(train, X), add(test, Xte)
    if llm_makebuy:
        X = add_llm_make_or_buy_features(train, X, "train")
        Xte = add_llm_make_or_buy_features(test, Xte, "test")
    return X, y, Xte[X.columns]


def e4_inputs(train, test, org_embed, concat_embed=False, e4_pca_dim=None):
    """E4(組織図エキスパート)の入力とモデル生成関数を返す。

    org_embed=True で TF-IDF+LR → OpenAI embedding+LR に **差し替える**（追加ではない）。
    入力を「文字列の1次元配列」から「埋め込みの2次元行列」に替えるだけで、
    fold スライス `e4[tr]` も fit も predict もそのまま動く。

    concat_embed=True で さらに 企業概要の埋め込みを横に連結する（H26 / exp029）。
    **7本目を足すのではなく E4 の中身を差し替える**のが要点。exp020/023/025 の
    3連敗は「既存の構成員に列を足した」形で、exp026 の ACCEPT は「別の情報源に
    差し替えた」形だった。企業概要は E0 も E3 も E6 も見ていない未使用の列なので、
    6本のまま新しい情報を入れられる。

    **E0 は触らない**。E0 が内部で持つ `組織図_購入確率` は現行の TF-IDF のまま
    にしてあるので、E0 の予測は org_embed の有無で完全一致する。これにより
    ホールドアウト検証で rep 単位の厳密なペアが取れる（`--survey-step` と同じ設計）。
    """
    if concat_embed:
        tr, te = load_concat_embeddings(CONCAT_SLUGS, dim=CONCAT_EMB_DIM,
                                        pca_dim=e4_pca_dim)
        params = {**EMB_LR_PARAMS, "C": CONCAT_EMB_C}
        return tr, te, lambda: build_embed_model(params)
    if org_embed:
        tr, te = load_embeddings("org", dim=ORG_EMB_DIM)
        params = {**EMB_LR_PARAMS, "C": ORG_EMB_C}
        return tr, te, lambda: build_embed_model(params)
    return load_org_text(train), load_org_text(test), build_org_model


def h30_inputs(enabled):
    """H30絶対差エキスパートの入力とモデル生成関数を返す。"""
    if not enabled:
        return None
    tr, te = load_absdiff_embeddings(dim=H30_ABSDIFF_DIM)
    params = {**EMB_LR_PARAMS, "C": H30_ABSDIFF_C}
    return tr, te, lambda: build_embed_model(params)


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


def _cross_lr(Xtr, ytr, Xva, Xte):
    """E7 の線形版。比率の裾が重いので中央値埋め＋標準化してから LR にかける。"""
    med = Xtr.median()
    f = lambda d: d.fillna(med).fillna(0)  # noqa: E731
    model = make_pipeline(StandardScaler(), LogisticRegression(**CROSS_LR))
    model.fit(f(Xtr), ytr)
    return (model.predict_proba(f(Xva))[:, 1], model.predict_proba(f(Xte))[:, 1])


def _onehot_align(*frames):
    """カテゴリ列を one-hot 化し、全フレームで列を揃える。

    `preprocess` が train のカテゴリ水準を test にも固定しているので get_dummies の
    出力列は本来一致するが、水準がゼロ件になる fold のために reindex で保険をかける。
    """
    outs = [pd.get_dummies(f, columns=[c for c in CAT_COLS if c in f.columns],
                           dummy_na=False)
            for f in frames]
    cols = outs[0].columns
    return [o.reindex(columns=cols, fill_value=0).astype(float) for o in outs]


E0B_NUMERIC_TRANSFORMS = ("raw", "signed_log")
E0B_SIGNED_LOG_COLS = FINANCE_COLS + SCALE_COLS


def _transform_e0b_numeric(frame, numeric_transform):
    """E0bの対象連続列へtarget-freeな決定的変換を適用する。"""
    if numeric_transform not in E0B_NUMERIC_TRANSFORMS:
        raise ValueError(f"未知のE0b数値前処理: {numeric_transform}")
    if numeric_transform == "raw":
        return frame
    out = frame.copy()
    for c in E0B_SIGNED_LOG_COLS:
        if c in out.columns:
            v = out[c].astype(float)
            out[c] = np.sign(v) * np.log1p(np.abs(v))
    return out


def fit_e0b(A_tr, ytr, A_va, A_te, C=LINEAR_C,
            numeric_transform="raw"):
    """E0b(H28): E0 と同じ94列に one-hot + 標準化 + LR。

    **列は減らさない**。田中さん nb018 は「線形向き54列に絞る」を試して 0/10 で負けており、
    主因は規模系30列（従業員数・売上・総資産等）を落としたこと。裾が重くても
    標準化 + 強めのL2で扱えている。分位変換も逆効果だった（比率の生の目盛りに情報がある）。

    欠損は **A_tr の中央値**で埋める（`資本金`/`営業利益`/`経常利益` は NaN のまま
    渡ってくる）。`_isna` フラグが別列にあるので、埋める定数自体はフラグの係数に
    吸収される＝値の選択は本質ではなく、標準化のスケールを壊さないための処置。

    numeric_transform="signed_log"はH61実験用。財務金額・規模23列だけに
    sign(x)*log1p(abs(x))を適用する。既定rawは過去予測と完全一致する。
    """
    Btr, Bva, Bte = _onehot_align(A_tr, A_va, A_te)
    Btr, Bva, Bte = [_transform_e0b_numeric(d, numeric_transform)
                      for d in (Btr, Bva, Bte)]
    med = Btr.median()
    f = lambda d: d.fillna(med).fillna(0.0)  # noqa: E731
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, **LINEAR_LR))
    model.fit(f(Btr), ytr)
    return (model.predict_proba(f(Bva))[:, 1], model.predict_proba(f(Bte))[:, 1])


def fit_e7(Xtr, ytr, Xva, yva, Xte, seed, e7_model="lgbm"):
    """E7(不満×財務)の学習。モデル族は診断で選ぶので切り替え可能にしてある。"""
    if e7_model == "lr":
        return _cross_lr(Xtr, ytr, Xva, Xte)
    return _lgbm(Xtr, ytr, Xva, yva, Xte, seed)


def compute_expert_preds(seed, cache=True, finance_ratios=False,
                         survey_step=False, org_embed=False,
                         concat_embed=False, cross=False, e7_model="lgbm",
                         linear=False, linear_c=LINEAR_C,
                         h30_absdiff=False, gam=False, gam_c=GAM_C,
                         embed_knn=False, h37_priority=False,
                         llm_e0_replace_dx_manual=False,
                         llm_e0_core4_pruned=False,
                         llm_e0_core4_pruned_e0_only=False,
                         llm_e0_core4_pruned_e0_only_h2=False,
                         llm_e0_core4_pruned_e0_only_h2_h21=False,
                         llm_e0_core4_pruned_e0_only_h2_both4=False,
                         e0_restore_missing_flags=False,
                         h21_anchor_both=False, h49_e0b_only=False,
                         e1_h54_prune=False, h2h7_both4=False,
                         h2_stable_leverage=False,
                         linear_preprocess="raw",
                         linear_log_extra=False, h58=False,
                         e4_pca_dim=None):
    """各エキスパートの OOF と test 予測を返す。重いのでnpzにキャッシュする。

    キャッシュ名に構成タグを含める。専用列のあり/なしで中身が変わるので、
    タグを分けないと片方の結果をもう片方に使い回してしまう。
    """
    if linear_log_extra and (not linear or linear_preprocess != "raw"):
        raise ValueError("linear_log_extraはraw E0bとの併用時だけ指定できる")
    names = expert_names(cross, linear, h30_absdiff, gam, embed_knn,
                         linear_log_extra)
    tag = (("_ratio" if finance_ratios else "") + ("_step" if survey_step else "")
           + ("_concatemb" if concat_embed else "_orgemb" if org_embed else "")
           + (("_cross" + ("lr" if e7_model == "lr" else "")) if cross else "")
           + (f"_lin{linear_c:g}" if linear else "")
           + (f"_linprep-{linear_preprocess}" if linear and
              linear_preprocess != "raw" else "")
           + ("_linlogextra" if linear_log_extra else "")
           + ("_h30abs" if h30_absdiff else "")
           + (f"_gam{gam_c:g}" if gam else "")
           + ("_h32knn" if embed_knn else "")
           + ("_h37" if h37_priority else "")
           + ("_llme0dx" if llm_e0_replace_dx_manual else "")
           + ("_llmcore4pruned" if llm_e0_core4_pruned else "")
           + ("_llmcore4e0only" if llm_e0_core4_pruned_e0_only else "")
           + ("_llmcore4e0onlyh2" if llm_e0_core4_pruned_e0_only_h2 else "")
           + ("_llmcore4e0onlyh2h21" if llm_e0_core4_pruned_e0_only_h2_h21 else "")
           + ("_llmcore4e0onlyh2both4v2eq" if llm_e0_core4_pruned_e0_only_h2_both4 else "")
           + ("_e0missing" if e0_restore_missing_flags else "")
           + ("_h21both" if h21_anchor_both else "")
           + ("_h49e0b" if h49_e0b_only else "")
           + ("_h54prune" if e1_h54_prune else "")
           + ("_h2h7both4v2eq" if h2h7_both4 else "")
           + ("_h2stablelevv2eq" if h2_stable_leverage else "")
           + ("_h58" if h58 else "")
           + (f"_e4pca{e4_pca_dim}" if e4_pca_dim else ""))
    path = os.path.join(OUT_DIR, f"_experts_seed{seed}{tag}.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        print(f"(キャッシュを利用: {path})")
        return d["y"], {n: d[f"oof_{n}"] for n in names}, \
            {n: d[f"te_{n}"] for n in names}

    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, finance_ratios=finance_ratios,
                               survey_step=survey_step, cross=cross,
                               h37_priority=h37_priority, h58=h58,
                               llm_makebuy=(llm_e0_replace_dx_manual or
                                            llm_e0_core4_pruned or
                                            llm_e0_core4_pruned_e0_only or
                                            llm_e0_core4_pruned_e0_only_h2 or
                                            llm_e0_core4_pruned_e0_only_h2_h21 or
                                            llm_e0_core4_pruned_e0_only_h2_both4 or
                                            h2h7_both4 or h2_stable_leverage),
                               h2_restore=(llm_e0_core4_pruned_e0_only_h2 or
                                           llm_e0_core4_pruned_e0_only_h2_h21 or
                                           llm_e0_core4_pruned_e0_only_h2_both4 or
                                           h2h7_both4 or h2_stable_leverage),
                               h21_restore=(llm_e0_core4_pruned_e0_only_h2_h21 or
                                            h21_anchor_both),
                               h49_restore=h49_e0b_only,
                               h7_restore=h2h7_both4)
    txt, txt_te = (train[TEXT_COL].fillna("").astype(str).values,
                   test[TEXT_COL].fillna("").astype(str).values)
    org, org_te = load_org_text(train), load_org_text(test)  # E0 が使う（常にTF-IDF）
    e4, e4_te, e4_model = e4_inputs(train, test, org_embed, concat_embed,
                                    e4_pca_dim=e4_pca_dim)
    h30 = h30_inputs(h30_absdiff)
    if h30 is not None:
        h30_in, h30_te, h30_model = h30
    if embed_knn:
        knn_in, knn_te = load_knn_embeddings()
    c0_all = e0_anchor_cols(X.columns)
    c0 = list(c0_all)
    if llm_e0_replace_dx_manual:
        # H14のDX辞書だけをLLM意味軸へ置換。nested TF-IDFスコアは残す。
        c0 = [c for c in c0 if not c.startswith("DX展望_")]
    if (llm_e0_core4_pruned or llm_e0_core4_pruned_e0_only or
            llm_e0_core4_pruned_e0_only_h2 or
            llm_e0_core4_pruned_e0_only_h2_h21 or
            llm_e0_core4_pruned_e0_only_h2_both4 or h2h7_both4 or
            h2_stable_leverage):
        missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
        core4 = {
            "LLM_need_score", "LLM_make_limitation_score",
            "LLM_buy_openness_score", "LLM_internal_training_capability_score",
        }
        c0 = [c for c in c0
              if not c.startswith("DX展望_") and c not in missing
              and (not c.startswith("LLM_") or c in core4)]
        if e0_restore_missing_flags:
            c0 += [c for c in c0_all if c in missing]
    if llm_e0_core4_pruned_e0_only_h2_h21:
        c0 += ["ソフト投資比率_売上", "流動比率"]
    if h21_anchor_both:
        c0 += ["ソフト投資比率_売上", "流動比率"]
    if h49_e0b_only:
        c0 = [c for c in c0 if c not in READINESS_COLS]
    e0_only = (llm_e0_core4_pruned_e0_only or llm_e0_core4_pruned_e0_only_h2 or
               llm_e0_core4_pruned_e0_only_h2_h21 or
               llm_e0_core4_pruned_e0_only_h2_both4 or h2h7_both4 or
               h2_stable_leverage)
    if llm_e0_core4_pruned_e0_only_h2_both4 or h2h7_both4 or h2_stable_leverage:
        c0b = [c for c in c0_all if not c.startswith("LLM_")]
    else:
        c0b = ([c for c in c0_all
                if not c.startswith("LLM_") and c not in H2_RESTORE_COLS
                and c not in FINANCE_RATIO_COLS]
               if e0_only else c0)
    if h21_anchor_both:
        c0b += ["ソフト投資比率_売上", "流動比率"]
    h2_mode = ("h2h7_both4" if h2h7_both4 else
               "h2_stable_leverage" if h2_stable_leverage else
               "h2_both4" if llm_e0_core4_pruned_e0_only_h2_both4 else None)
    if h2_mode is not None:
        c0, c0b = route_h2_anchor_cols(c0, c0b, h2_mode)
    if h49_e0b_only:
        c0b = [c for c in c0b if c not in READINESS_COLS]
        c0b += ["展開段階_移行幅", "展開段階_scale_window"]
    c1, c2, c6 = (e1_finance_cols(with_ratios=finance_ratios,
                                  h54_prune=e1_h54_prune, with_h58=h58),
                  e2_survey_cols(with_step=survey_step),
                  e6_manual_cols(X.columns))
    c7 = e7_cross_cols() if cross else None

    oof = {n: np.zeros(len(X)) for n in names}
    te = {n: [] for n in names}
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

        if h30 is not None:
            m_h30 = h30_model().fit(h30_in[tr], y[tr])
            oof[E8_NAME][va] = m_h30.predict_proba(h30_in[va])[:, 1]
            te[E8_NAME].append(m_h30.predict_proba(h30_te)[:, 1])

        if embed_knn:
            m_knn = build_knn_model().fit(knn_in[tr], y[tr])
            oof[E9_NAME][va] = m_knn.predict_proba(knn_in[va])[:, 1]
            te[E9_NAME].append(m_knn.predict_proba(knn_te)[:, 1])

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

        if linear:
            # E0 と同一の94列（fold内テキストスコア込み）に one-hot + LR
            if c0b == c0:
                B_tr, B_va, B_te = A_tr, A_va, A_te
            else:
                B_tr, B_va, B_te = (Xtr[c0b].copy(), Xva[c0b].copy(),
                                     Xte[c0b].copy())
                for col, v in ((DX_SCORE_COL, (t_tr, t_va, t_te)),
                               (ORG_SCORE_COL, (o_tr, o_va, o_te))):
                    B_tr[col], B_va[col], B_te[col] = v
            p, q = fit_e0b(B_tr, y[tr], B_va, B_te, linear_c,
                           linear_preprocess)
            oof[E0B_NAME][va] = p
            te[E0B_NAME].append(q)
            if linear_log_extra:
                p, q = fit_e0b(B_tr, y[tr], B_va, B_te, linear_c,
                               "signed_log")
                oof[E0B_LOG_NAME][va] = p
                te[E0B_LOG_NAME].append(q)

        if gam:
            p, q = fit_gam(A_tr, y[tr], A_va, A_te, C=gam_c)
            oof[GAM_NAME][va] = p
            te[GAM_NAME].append(q)

        for name, cols in (("E1_finance", c1), ("E6_manual", c6)):
            p, q = _lgbm(Xtr[cols], y[tr], Xva[cols], y[va], Xte[cols], seed)
            oof[name][va] = p
            te[name].append(q)

        p, q = _survey_lr(Xtr[c2], y[tr], Xva[c2], Xte[c2])
        oof["E2_survey"][va] = p
        te["E2_survey"].append(q)

        if cross:
            p, q = fit_e7(Xtr[c7], y[tr], Xva[c7], y[va], Xte[c7], seed,
                          e7_model)
            oof[E7_NAME][va] = p
            te[E7_NAME].append(q)

    te = {n: np.mean(v, axis=0) for n, v in te.items()}
    np.savez(path, y=y, **{f"oof_{n}": oof[n] for n in names},
             **{f"te_{n}": te[n] for n in names})
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


def blend_oof(y, oof, te, seed, alpha=None, experts=EXPERTS, target="zero",
              alpha_rule="auto"):
    """メタ学習器を外側fold内でfitして、合成OOFとtest確率を返す。

    alpha=None なら **外側trainの中だけで** alpha を選ぶ。foldごとに選び直すので、
    評価対象の外側validはalpha選択にも重み推定にも一切寄与しない。

    alpha_rule="1se" にすると、選ぶ範囲は同じだが「最良から1標準誤差以内で
    最小の alpha」に留める。内側AP曲線は 0.001〜0.03 でほぼ平ら（本番構成
    11seed×5foldで幅0.0015）なので、argmax はノイズを拾っており、たまたま
    大きい alpha を引くと直交している E2 の重みが潰れる。
    詳細は documents/meta_native_and_alpha_selection_20260829.md。

    experts で構成員を絞れる。キャッシュには常に全エキスパートの予測が入っているので、
    部分集合の評価に再計算は要らない。
    """
    Z = to_logit(np.column_stack([oof[n] for n in experts]))
    Zte = to_logit(np.column_stack([te[n] for n in experts]))
    out = np.zeros(len(y))
    weights, te_parts, alphas = [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in skf.split(Z, y):
        if alpha is not None:
            a = alpha
        elif alpha_rule == "1se":
            a = select_alpha_1se(Z[tr], y[tr], seed, target=target)
        else:
            a = select_alpha(Z[tr], y[tr], seed=seed, target=target)[0]
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


def run(seed, alpha, submit, no_cache, experts=None, finance_ratios=False,
        test_pos=None, threshold=None, target="zero", survey_step=False,
        org_embed=False, concat_embed=False, cross=False, e7_model="lgbm",
        linear=False, linear_c=LINEAR_C, h30_absdiff=False,
        gam=False, gam_c=GAM_C, embed_knn=False, h37_priority=False,
        llm_e0_replace_dx_manual=False, llm_e0_core4_pruned=False,
        llm_e0_core4_pruned_e0_only=False,
        llm_e0_core4_pruned_e0_only_h2=False,
        llm_e0_core4_pruned_e0_only_h2_h21=False,
        llm_e0_core4_pruned_e0_only_h2_both4=False,
        e0_restore_missing_flags=False, h21_anchor_both=False,
        h49_e0b_only=False, e1_h54_prune=False, h2h7_both4=False,
        h2_stable_leverage=False, output=None,
        linear_preprocess="raw", linear_log_extra=False, h58=False,
        e4_pca_dim=None, alpha_rule="auto"):
    default = expert_names(cross, linear, h30_absdiff, gam, embed_knn,
                           linear_log_extra)
    experts = default if experts is None else experts
    y, oof, te = compute_expert_preds(seed, cache=not no_cache,
                                      finance_ratios=finance_ratios,
                                      survey_step=survey_step,
                                      org_embed=org_embed,
                                      concat_embed=concat_embed,
                                      cross=cross, e7_model=e7_model,
                                      linear=linear, linear_c=linear_c,
                                      h30_absdiff=h30_absdiff,
                                      gam=gam, gam_c=gam_c,
                                      embed_knn=embed_knn,
                                      h37_priority=h37_priority,
                                      llm_e0_replace_dx_manual=llm_e0_replace_dx_manual,
                                      llm_e0_core4_pruned=llm_e0_core4_pruned,
                                      llm_e0_core4_pruned_e0_only=llm_e0_core4_pruned_e0_only,
                                      llm_e0_core4_pruned_e0_only_h2=llm_e0_core4_pruned_e0_only_h2,
                                      llm_e0_core4_pruned_e0_only_h2_h21=llm_e0_core4_pruned_e0_only_h2_h21,
                                      llm_e0_core4_pruned_e0_only_h2_both4=llm_e0_core4_pruned_e0_only_h2_both4,
                                      e0_restore_missing_flags=e0_restore_missing_flags,
                                      h21_anchor_both=h21_anchor_both,
                                      h49_e0b_only=h49_e0b_only,
                                      e1_h54_prune=e1_h54_prune,
                                      h2h7_both4=h2h7_both4,
                                      h2_stable_leverage=h2_stable_leverage,
                                      linear_preprocess=linear_preprocess,
                                      linear_log_extra=linear_log_extra,
                                      h58=h58, e4_pca_dim=e4_pca_dim)
    if target != "zero":
        print(f"メタ重みの縮小先: {target}")
    if finance_ratios:
        print(f"E1に H21 財務比{5 if h58 else 4}本"
              f"{'(H58込み)' if h58 else ''}を追加した構成")
    if survey_step:
        print("E2に H22 閾値指標4本を追加した構成")
    if cross:
        print(f"E7(不満×財務 交互作用12列 / H27, model={e7_model})を7本目に追加した構成")
    if linear:
        print(f"E0b(E0と同じ94列に one-hot+LR / H28, C={linear_c:g}, "
              f"数値前処理={linear_preprocess})を追加した構成")
    if linear_log_extra:
        print("raw E0bを維持し、符号付き対数E0bを追加した二重線形構成")
    if h30_absdiff:
        print(f"E8(H30 3文書間絶対差embedding, dim={H30_ABSDIFF_DIM}, "
              f"C={H30_ABSDIFF_C:g})を追加した構成")
    if gam:
        print(f"E0c(H31 同じ94列の低自由度GAM, C={gam_c:g})を追加した構成")
    if embed_knn:
        print(f"E9(H32 [{' ; '.join(KNN_SLUGS)}] embedding cosine-kNN, "
              f"dim={KNN_DIM}, k={KNN_NEIGHBORS}, scale={KNN_DISTANCE_SCALE:g}, "
              f"prior={KNN_PRIOR_STRENGTH:g})を追加した構成")
    if h37_priority:
        print("H37 governance/priority 2列を E0/E0b に追加したexp008復元構成")
    if llm_e0_replace_dx_manual:
        print(f"E0/E0bのDX手動辞書をLLM Make-or-Buy {len(LLM_FEATURE_COLS)}列へ置換")
    if llm_e0_core4_pruned:
        print("E0/E0bからDX辞書22列・欠損フラグ7列を削除しLLM親スコア4列へ置換")
    if llm_e0_core4_pruned_e0_only:
        print("E0だけDX辞書・欠損flagsを削りLLM親4軸へ置換（E0bは元の94列）")
    if llm_e0_core4_pruned_e0_only_h2:
        print("E0だけpruned+LLM親4軸へ置換し、H2（自己資本比率・ROA）を復元")
    if llm_e0_core4_pruned_e0_only_h2_h21:
        print("E0だけpruned+LLM親4軸へ置換し、H2＋H21の財務4列を復元")
    if llm_e0_core4_pruned_e0_only_h2_both4:
        print("E0へH2財務健全性4列を追加し、E0bにも同じ4列を共有")
    if e0_restore_missing_flags:
        print("削減E0へ財務・アンケート欠損フラグ7列を復元")
    if h21_anchor_both:
        print("H21（ソフト投資/売上・流動比率）をE0とE0bの両方へ追加")
    if h49_e0b_only:
        print("H49展開段階2列をE0bだけへ追加")
    if e1_h54_prune:
        print("E1のH54 target-free会計恒等式重複5本を削除(19->14列)")
    if h2h7_both4:
        print("H2財務4本の負債比率/ROEをFCF比率/現金ROAへ差替(H2+H7)")
    if h2_stable_leverage:
        print("E0bだけ負債比率を負債資産比率へ差替(E0は現状維持)")
    if concat_embed:
        print(f"E4 を [{' ; '.join(CONCAT_SLUGS)}] 連結 embedding+LR に差し替えた構成 "
              f"(ブロックごと dim={CONCAT_EMB_DIM}, C={CONCAT_EMB_C})")
    elif org_embed:
        print(f"E4 を組織図 embedding+LR に差し替えた構成 "
              f"(dim={ORG_EMB_DIM}, C={ORG_EMB_C})")
    if tuple(experts) != default:
        print(f"構成員: {list(experts)}")

    rows = {n: _scores(y, oof[n]) for n in experts}
    Z = to_logit(np.column_stack([oof[n] for n in experts]))
    rows["-- uniform blend"] = _scores(y, uniform_blend(Z))
    blend, blend_te, w, alphas = blend_oof(y, oof, te, seed, alpha,
                                           experts=experts, target=target,
                                           alpha_rule=alpha_rule)
    key = (f"-- meta (alpha={alpha_rule})" if alpha is None
           else f"-- meta (alpha={alpha})")
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
        tag = alpha_rule if alpha is None else str(alpha)
        n_tag = "" if tuple(experts) == default else f"_{len(experts)}experts"
        r_tag = (("_ratio" if finance_ratios else "")
                 + ("_step" if survey_step else "")
                 + ("_concatemb" if concat_embed else
                    "_orgemb" if org_embed else "")
                 + (("_cross" + ("lr" if e7_model == "lr" else ""))
                    if cross else "")
                 + (f"_lin{linear_c:g}" if linear else "")
                 + (f"_linprep-{linear_preprocess}" if linear and
                    linear_preprocess != "raw" else "")
                 + ("_linlogextra" if linear_log_extra else "")
                 + ("_h30abs" if h30_absdiff else "")
                 + (f"_gam{gam_c:g}" if gam else "")
                 + ("_h32knn" if embed_knn else "")
                 + ("_h37" if h37_priority else "")
                 + ("_llme0dx" if llm_e0_replace_dx_manual else "")
                 + ("_llmcore4pruned" if llm_e0_core4_pruned else "")
                 + ("_llmcore4e0only" if llm_e0_core4_pruned_e0_only else "")
                 + ("_llmcore4e0onlyh2" if llm_e0_core4_pruned_e0_only_h2 else "")
                 + ("_llmcore4e0onlyh2h21" if llm_e0_core4_pruned_e0_only_h2_h21 else "")
                 + ("_llmcore4e0onlyh2both4v2eq" if llm_e0_core4_pruned_e0_only_h2_both4 else "")
                 + ("_e0missing" if e0_restore_missing_flags else "")
                 + ("_h21both" if h21_anchor_both else "")
                 + ("_h49e0b" if h49_e0b_only else "")
                 + ("_h54prune" if e1_h54_prune else "")
                 + ("_h2h7both4v2eq" if h2h7_both4 else "")
                 + ("_h2stablelevv2eq" if h2_stable_leverage else "")
                 + ("_h58" if h58 else "")
                 + (f"_e4pca{e4_pca_dim}" if e4_pca_dim else ""))
        t_tag = (f"_pos{test_pos}" if test_pos is not None
                 else f"_th{threshold}" if threshold is not None else "")
        out = output or (f"submission/submission_ensemble_experts_seed{seed}"
                         f"_a{tag}{n_tag}{r_tag}{t_tag}.csv")
        if os.path.exists(out):
            raise FileExistsError(f"既存の提出物は上書きしない: {out}")
        sub.assign(pred=sub["pred"].astype(int)).to_csv(
            out, index=False, header=False, lineterminator="\n")
        print(f"保存: {out} 正例={int(label.sum())} (th={th:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=None,
                   help="メタのL2強度。省略時は外側trainの中だけで自動選択")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--output",
                   help="提出CSVの保存先。既存ファイルは上書きしない")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--drop", nargs="*", default=[],
                   choices=list(EXPERTS) + [E7_NAME, E0B_NAME, E0B_LOG_NAME,
                                            E8_NAME, GAM_NAME, E9_NAME],
                   help="構成員から外すエキスパート（キャッシュは全員分あるので再計算不要）")
    p.add_argument("--finance-ratios", action="store_true",
                   help="E1 に H21 財務比4本を足す（E0は現行94列のまま）")
    p.add_argument("--e4-pca-dim", type=int, default=None,
                   help="E4の連結埋め込みを全3072次元からPCAでこの次元へ畳む"
                        "（--concat-embed と併用。既定Noneで従来のMRL切り詰め）")
    p.add_argument("--h58", action="store_true",
                   help="H58(無形投資_減価償却比)を財務比の5本目として足す"
                        "（--finance-ratios と併用。単独指定は無効）")
    p.add_argument("--survey-step", action="store_true",
                   help="E2 に H22 閾値指標4本を足す（E0は現行94列のまま）")
    p.add_argument("--org-embed", action="store_true",
                   help="E4 を組織図の OpenAI embedding+LR に差し替える（E0は不変）")
    p.add_argument("--concat-embed", action="store_true",
                   help="E4 を [組織図 ; 企業概要] 連結 embedding+LR に差し替える"
                        "（--org-embed より優先。E0は不変）")
    p.add_argument("--cross", action="store_true",
                   help="H27 の不満×財務12列で E7 を立て、7本目に足す（既存6本は不変）")
    p.add_argument("--e7-model", default="lgbm", choices=["lgbm", "lr"],
                   help="E7 のモデル族。診断で選ぶ（E2 は LR が LGBM に明確勝ちだった）")
    p.add_argument("--linear", action="store_true",
                   help="H28 の E0b(E0と同じ94列に one-hot+LR)を足す（既存は不変）")
    p.add_argument("--linear-c", type=float, default=LINEAR_C,
                   help="E0b の LR の C（既定 0.1 ＝ 田中さん nb018 の設定）")
    p.add_argument("--linear-preprocess", choices=E0B_NUMERIC_TRANSFORMS,
                   default="raw",
                   help="E0bの金額・規模列の前処理。signed_logは符号付きlog1p")
    p.add_argument("--linear-log-extra", action="store_true",
                   help="raw E0bを残し、符号付きlog1p版E0bを追加Expertにする")
    p.add_argument("--h30-absdiff", action="store_true",
                   help="H30の3文書間絶対差embeddingを独立エキスパートとして追加")
    p.add_argument("--gam", action="store_true",
                   help="H31の低自由度GAMを追加（--drop E0b_linear で置換比較）")
    p.add_argument("--gam-c", type=float, default=GAM_C,
                   help="H31 GAMのLR正則化C（事前固定値0.01）")
    p.add_argument("--embed-knn", action="store_true",
                   help="H32のDX展望+組織図embedding cosine-kNNを追加")
    p.add_argument("--h37-priority", action="store_true",
                   help="H37 governance/priority 2列をE0/E0bへ追加（exp008復元）")
    p.add_argument("--llm-e0-replace-dx-manual", action="store_true",
                   help="E0/E0b内のDX手動辞書をLLM Make-or-Buy特徴へ置換")
    p.add_argument("--llm-e0-core4-pruned", action="store_true",
                   help="E0/E0bのDX辞書・欠損flagsを削りLLM親スコア4列へ置換")
    p.add_argument("--llm-e0-core4-pruned-e0-only", action="store_true",
                   help="E0だけpruned+LLM親4軸へ置換し、E0bは元の94列を維持")
    p.add_argument("--llm-e0-core4-pruned-e0-only-h2", action="store_true",
                   help="E0-only pruned構成へH2（自己資本比率・ROA）だけ復元")
    p.add_argument("--llm-e0-core4-pruned-e0-only-h2-h21", action="store_true",
                   help="E0-only pruned構成へH2＋H21の財務4列を復元")
    p.add_argument("--llm-e0-core4-pruned-e0-only-h2-both4", action="store_true",
                   help="E0へH2財務健全性4列を追加し、E0bにも4列すべて共有")
    p.add_argument("--e0-restore-missing-flags", action="store_true",
                   help="削減E0へ財務・アンケート欠損フラグ7列を復元")
    p.add_argument("--h21-anchor-both", action="store_true",
                   help="H21代表2列をE0とE0bの両方へ追加")
    p.add_argument("--h49-e0b-only", action="store_true",
                   help="H49展開段階2列をE0bだけへ追加")
    p.add_argument("--e1-h54-prune", action="store_true",
                   help="E1のH54 target-free会計恒等式重複5本を削除(19->14列)")
    p.add_argument("--h2h7-both4", action="store_true",
                   help="H2財務4本の負債比率/ROEをFCF比率/現金ROAへ差替(legacy_batch1で5/5満点だった組合せ)")
    p.add_argument("--h2-stable-leverage", action="store_true",
                   help="E0bだけ負債比率(不安定)を負債資産比率(安定)へ差替、E0は現状維持")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--test-pos", type=int,
                   help="testの予測正例数を指定して閾値を逆算（既定はOOF最適）")
    g.add_argument("--threshold", type=float, help="閾値を直接指定")
    p.add_argument("--alpha-rule", default="auto", choices=["auto", "1se"],
                   help="メタのalpha選択則。1se は最良から1SE以内で最小のalphaに"
                        "留める（既定 auto は現行のargmax）")
    p.add_argument("--meta-target", default="zero", choices=["zero", "uniform"],
                   help="メタ重みの縮小先。uniform は等分ブレンドへ縮む")
    a = p.parse_args()
    run(a.seed, a.alpha, a.submit, a.no_cache,
        experts=tuple(n for n in expert_names(a.cross, a.linear, a.h30_absdiff,
                                              a.gam, a.embed_knn,
                                              a.linear_log_extra)
                      if n not in a.drop),
        finance_ratios=a.finance_ratios, h58=a.h58, e4_pca_dim=a.e4_pca_dim,
        test_pos=a.test_pos, threshold=a.threshold, target=a.meta_target,
        survey_step=a.survey_step, org_embed=a.org_embed,
        concat_embed=a.concat_embed, cross=a.cross, e7_model=a.e7_model,
        linear=a.linear, linear_c=a.linear_c,
        h30_absdiff=a.h30_absdiff, gam=a.gam, gam_c=a.gam_c,
        embed_knn=a.embed_knn, h37_priority=a.h37_priority,
        llm_e0_replace_dx_manual=a.llm_e0_replace_dx_manual,
        llm_e0_core4_pruned=a.llm_e0_core4_pruned,
        llm_e0_core4_pruned_e0_only=a.llm_e0_core4_pruned_e0_only,
        llm_e0_core4_pruned_e0_only_h2=a.llm_e0_core4_pruned_e0_only_h2,
        llm_e0_core4_pruned_e0_only_h2_h21=a.llm_e0_core4_pruned_e0_only_h2_h21,
        llm_e0_core4_pruned_e0_only_h2_both4=a.llm_e0_core4_pruned_e0_only_h2_both4,
        e0_restore_missing_flags=a.e0_restore_missing_flags,
        h21_anchor_both=a.h21_anchor_both,
        h49_e0b_only=a.h49_e0b_only,
        e1_h54_prune=a.e1_h54_prune,
        h2h7_both4=a.h2h7_both4,
        h2_stable_leverage=a.h2_stable_leverage,
        output=a.output, linear_preprocess=a.linear_preprocess,
        linear_log_extra=a.linear_log_extra, alpha_rule=a.alpha_rule)
