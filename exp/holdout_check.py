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

B はどのエキスパートもメタも閾値決定も見ないため、ここでの差を本判定に使う。

注意（2026-08-26監査）:
  旧実装は上記の意図に反し、LightGBMのearly stoppingへBラベルを渡していた。
  現在は学習側の内部holdoutで木数を決めて全学習側へrefitするstrict方式が既定。
  過去結果の再現に限り ``--legacy-leaky-lgbm`` で旧方式を明示できるが、採否には使わない。

閾値則の比較（`--threshold-rules`）:
  exp020 は「順位付け(AUC/AP)は同等以上なのに閾値を当てられず F1 が 0/5」で
  REJECT された。閾値の決め方そのものが効いている可能性が高いので、同じ
  エキスパート予測に対して閾値則だけを差し替えて比較できるようにする。
  どの則も **A の情報だけ**で閾値を決める（rate は B のスコア分布だけを使い、
  B のラベルは見ない）。

  python exp/holdout_check.py [--n-reps 3] [--alpha 0.01]
                              [--threshold-rules oof_best fold_mean fold_median
                                                   plateau_mid rate half_f1]
"""
import argparse
import os
import sys

import lightgbm as lgb
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
    FINANCE_ISNA_COLS, SURVEY_ISNA_COLS, e0_anchor_cols, e1_finance_cols,
    e2_survey_cols, e6_manual_cols, e7_cross_cols, route_h2_anchor_cols,
)
from ensemble_experts import (  # noqa: E402
    DX_SCORE_COL, E0B_NAME, E7_NAME, E8_NAME, E9_NAME, EXPERTS, GAM_NAME, LINEAR_C,
    THS, _lgbm,
    _survey_lr, build_features, e4_inputs, expert_names, fit_e0b, fit_e7,
    h30_inputs,
)
from harness import BASE_PARAMS  # noqa: E402
from gam_model import GAM_C, fit_gam  # noqa: E402
from embedding_knn import (  # noqa: E402
    KNN_DIM, KNN_DISTANCE_SCALE, KNN_NEIGHBORS, KNN_PRIOR_STRENGTH,
    build_knn_model, load_knn_embeddings,
)
from decision import verdict  # noqa: E402
from meta_blend import (  # noqa: E402
    apply_meta, fit_meta, select_alpha, to_logit, uniform_blend,
)


def _strict_lgbm_predict(Xtr, ytr, Xtarget, seed):
    """target のラベルを使わずに木数を決め、Xtr 全体で再学習して予測する。

    従来の完全ホールドアウト実装は target=B のときにも ``y[B]`` を
    early stoppingへ渡していた。ここでは学習集合内の固定5分割の先頭foldだけを
    木数選択に使い、その木数で学習集合全体をfitし直す。したがって target は
    特徴量しか参照されず、A内OOFと最終B予測のどちらも評価先ラベルから独立になる。
    """
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, stop_idx = next(inner.split(Xtr, ytr))
    params = {**BASE_PARAMS, "seed": seed}
    scout = lgb.train(
        params,
        lgb.Dataset(Xtr.iloc[fit_idx], ytr[fit_idx]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xtr.iloc[stop_idx], ytr[stop_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    rounds = max(1, int(scout.best_iteration))
    final = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
    return final.predict(Xtarget, num_iteration=rounds)


def fit_predict_experts(X, y, txt, org, tr, target, seed, cols, e4=None,
                        e7_model=None, linear_c=None, h30=None, gam_c=None,
                        knn=None, strict_lgbm=True):
    """tr で全エキスパートを学習し、target 行を予測する。

    E0 のテキスト特徴は tr の内側CVで作る（tr 内で完結＝リーク無し）。
    target 側は tr 全体で学習したテキストモデルで付ける。

    e4=(入力, モデル生成関数) を渡すと E4 の実装を差し替える（--org-embed）。
    E0 は常に org(TF-IDF) を使うので、差し替えの有無で E0 の予測は完全一致し、
    rep 単位の厳密なペア比較ができる。

    e7_model を渡すと E7(H27 不満×財務) を7本目として学習する（--cross）。
    E7 の12列は EXPERT_ONLY_COLS なので E0 は見ておらず、こちらも E0 の予測は
    --cross の有無で完全一致する。
    """
    c0, c0b, c1, c2, c6, c7 = cols
    e4_in, e4_model = e4 if e4 is not None else (org, build_org_model)
    Xtr, Xta = X.iloc[tr], X.iloc[target]
    out = {}

    m_txt = build_model().fit(txt[tr], y[tr])
    out["E3_dx_text"] = m_txt.predict_proba(txt[target])[:, 1]
    m_org = e4_model().fit(e4_in[tr], y[tr])
    out["E4_org_text"] = m_org.predict_proba(e4_in[target])[:, 1]
    if h30 is not None:
        h30_in, h30_model = h30
        m_h30 = h30_model().fit(h30_in[tr], y[tr])
        out[E8_NAME] = m_h30.predict_proba(h30_in[target])[:, 1]
    if knn is not None:
        knn_in, knn_model = knn
        m_knn = knn_model().fit(knn_in[tr], y[tr])
        out[E9_NAME] = m_knn.predict_proba(knn_in[target])[:, 1]

    t_tr, t_ta = nested_text_pred(txt[tr], y[tr], txt[target])
    o_tr, o_ta, _ = fold_org_preds(org[tr], y[tr], org[target], org[:1])
    A_tr, A_ta = Xtr[c0].copy(), Xta[c0].copy()
    A_tr[DX_SCORE_COL], A_ta[DX_SCORE_COL] = t_tr, t_ta
    A_tr[ORG_SCORE_COL], A_ta[ORG_SCORE_COL] = o_tr, o_ta
    if strict_lgbm:
        out["E0_anchor"] = _strict_lgbm_predict(A_tr, y[tr], A_ta, seed)
    else:
        out["E0_anchor"], _ = _lgbm(A_tr, y[tr], A_ta, y[target], A_ta, seed)
    if linear_c is not None:
        B_tr, B_ta = Xtr[c0b].copy(), Xta[c0b].copy()
        B_tr[DX_SCORE_COL], B_ta[DX_SCORE_COL] = t_tr, t_ta
        B_tr[ORG_SCORE_COL], B_ta[ORG_SCORE_COL] = o_tr, o_ta
        out[E0B_NAME], _ = fit_e0b(B_tr, y[tr], B_ta, B_ta, linear_c)
    if gam_c is not None:
        out[GAM_NAME], _ = fit_gam(A_tr, y[tr], A_ta, A_ta, C=gam_c)

    for name, c in (("E1_finance", c1), ("E6_manual", c6)):
        if strict_lgbm:
            out[name] = _strict_lgbm_predict(Xtr[c], y[tr], Xta[c], seed)
        else:
            out[name], _ = _lgbm(Xtr[c], y[tr], Xta[c], y[target], Xta[c], seed)
    out["E2_survey"], _ = _survey_lr(Xtr[c2], y[tr], Xta[c2], Xta[c2])
    if e7_model is not None:
        if strict_lgbm and e7_model == "lgbm":
            out[E7_NAME] = _strict_lgbm_predict(Xtr[c7], y[tr], Xta[c7], seed)
        else:
            out[E7_NAME], _ = fit_e7(Xtr[c7], y[tr], Xta[c7], y[target],
                                     Xta[c7], seed, e7_model)
    return out


def subsets(cross=False, linear=False, h30_absdiff=False, gam=False,
            embed_knn=False):
    """比較する構成。基準は常に meta_6本。候補を後ろに足していく。

    cross/linear を両方立てると meta_7本(=+E7) と meta_8本(=+E7+E0b) が並ぶので、
    「E0b を足した効果」は meta_8本 - meta_7本 の対リピート差分で読む。
    """
    s = {
        "meta_6本": EXPERTS,
        "meta_4本": tuple(n for n in EXPERTS
                          if n not in ("E3_dx_text", "E4_org_text")),
    }
    if cross:
        s["meta_7本"] = expert_names(cross=True)
    if linear:
        s["meta_8本" if cross else "meta_7本_lin"] = expert_names(cross, True)
    if h30_absdiff:
        n_base = len(expert_names(cross, linear))
        s[f"meta_{n_base + 1}本"] = expert_names(cross, linear, True)
    if gam:
        without_linear = expert_names(cross, False, h30_absdiff, True)
        s[f"meta_{len(without_linear)}本_gam"] = without_linear
        if linear:
            both = expert_names(cross, True, h30_absdiff, True)
            s[f"meta_{len(both)}本_both"] = both
    if embed_knn:
        candidate = expert_names(cross, linear, h30_absdiff, gam, True)
        s[f"meta_{len(candidate)}本_knn"] = candidate
    return s


SUBSETS = subsets()


def _stack(preds, experts=EXPERTS):
    return to_logit(np.column_stack([preds[n] for n in experts]))


def _best_th(y, p):
    return THS[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in THS]))]


# 閾値則。すべて A の情報だけで決める（rate のみ B のスコア分布を使うがラベルは見ない）。
THRESHOLD_RULES = ("oof_best", "fold_mean", "fold_median", "plateau_mid",
                   "rate", "half_f1")


def _fold_curve_thresholds(pA, yA, folds):
    """foldごとのF1曲線、従来argmax、同率最大中央の閾値を返す。"""
    fold_curves, fold_argmax, fold_tie_mid = [], [], []
    for _, va in folds:
        curve = np.array([f1_score(yA[va], (pA[va] >= t).astype(int))
                          for t in THS])
        fold_curves.append(curve)
        fold_argmax.append(float(THS[int(np.argmax(curve))]))
        ties = THS[np.isclose(curve, curve.max(), rtol=0.0, atol=1e-12)]
        fold_tie_mid.append(float(np.median(ties)))
    return (np.vstack(fold_curves), np.asarray(fold_argmax),
            np.asarray(fold_tie_mid))


def _plateau_midpoint(fold_curves, tol=0.005):
    """fold平均F1曲線の最大点を含む連続プラトーの中央を返す。"""
    mean_curve = fold_curves.mean(axis=0)
    peak = int(np.argmax(mean_curve))
    inside = mean_curve >= mean_curve[peak] - tol
    left = peak
    while left > 0 and inside[left - 1]:
        left -= 1
    right = peak
    while right + 1 < len(inside) and inside[right + 1]:
        right += 1
    midpoint = (THS[left] + THS[right]) / 2.0
    return float(THS[int(np.argmin(np.abs(THS - midpoint)))])


def _thresholds(pA, yA, pB, folds, rules):
    """A の予測から閾値を決める。B のラベルは一切使わない。

    oof_best  A全体のOOFでF1最大（現行。argmaxを1点で取るので過学習しやすい）
    fold_mean A内5foldそれぞれのF1最大閾値の平均（平均化で分散を落とす）
    fold_median foldごとの最適閾値の中央値（外れfoldへの耐性を上げる）
    plateau_mid fold別F1曲線を平均し、最大点から0.005以内の連続な高原の中央。
                離れた別ピークは混ぜず、argmax一点への合わせ込みを抑える
    rate      A での予測正例率を B のスコア分位点に転写（rate matching）
    half_f1   達成F1の半分。校正済み確率のF1最適閾値は F1*/2 という既知の関係を
              使う。自由度が最小＝最も過学習しにくい
    """
    out = {}
    best = None
    if {"oof_best", "rate", "half_f1"} & set(rules):
        best = _best_th(yA, pA)
    fold_curves = fold_argmax = fold_tie_mid = None
    if {"fold_mean", "fold_median", "plateau_mid"} & set(rules):
        fold_curves, fold_argmax, fold_tie_mid = _fold_curve_thresholds(
            pA, yA, folds)
    for r in rules:
        if r == "oof_best":
            out[r] = float(best)
        elif r == "fold_mean":
            out[r] = float(fold_argmax.mean())
        elif r == "fold_median":
            out[r] = float(np.median(fold_tie_mid))
        elif r == "plateau_mid":
            out[r] = _plateau_midpoint(fold_curves)
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
            target="zero", e4=None, e7_model=None, linear_c=None, h30=None,
            gam_c=None, knn=None, strict_lgbm=True):
    idx = np.arange(len(y))
    A, B = train_test_split(idx, test_size=test_size, stratify=y, random_state=rep)
    cross, linear = e7_model is not None, linear_c is not None
    h30_absdiff = h30 is not None
    gam = gam_c is not None
    embed_knn = knn is not None
    names = expert_names(cross, linear, h30_absdiff, gam, embed_knn)
    subs = subsets(cross, linear, h30_absdiff, gam, embed_knn)

    # 2. A の中だけでエキスパートOOFを作る
    oof = {n: np.zeros(len(A)) for n in names}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
    folds = list(skf.split(A, y[A]))
    for tr, va in folds:
        p = fit_predict_experts(X, y, txt, org, A[tr], A[va], rep, cols, e4,
                                e7_model, linear_c, h30, gam_c, knn,
                                strict_lgbm)
        for n in names:
            oof[n][va] = p[n]

    # 3. A のOOFで alpha と重みを決める（Bは一切見ない）
    yA = y[A]
    fitted, alphas = {}, {}
    for label, members in subs.items():
        Za = _stack(oof, members)
        a = (alpha if alpha is not None
             else select_alpha(Za, yA, seed=rep, target=target)[0])
        w, b = fit_meta(Za, yA, alpha=a, target=target)
        fitted[label] = (members, w, b)
        alphas[label] = a

    # 4. A全体で再学習 -> B を予測
    pb = fit_predict_experts(X, y, txt, org, A, B, rep, cols, e4, e7_model,
                             linear_c, h30, gam_c, knn, strict_lgbm)
    yB = y[B]

    # A側/B側のスコアを同じ形で並べる（閾値はAから、評価はBで）
    # エキスパート単体も入れる: 合成後に改善が消えるのが「E1自体が弱るから」なのか
    # 「メタが薄めるから」なのかを切り分けるため。
    pairs = {n: (oof[n], pb[n]) for n in names}
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
        target="zero", survey_step=False, org_embed=False, concat_embed=False,
        cross=False, e7_model="lgbm", linear=False, linear_c=LINEAR_C,
        h30_absdiff=False, gam=False, gam_c=GAM_C, embed_knn=False,
        out_csv=None, llm_e0_core4_pruned=False,
        llm_e0_core4_pruned_e0_only_h2_both4=False, e6_drop_dx_manual=False,
        e6_add_llm4=False, e1_add_h2=False, e6_h52_prune=False,
        e1_h54_prune=False, e6_add_noise_col=False, seed_offset=0,
        e2_h57_subscales=False, h2h7_both4=False, h2_stable_leverage=False,
        strict_lgbm=True):
    train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    X, y, _ = build_features(train, test, finance_ratios=finance_ratios,
                             survey_step=survey_step, cross=cross,
                             llm_makebuy=(llm_e0_core4_pruned or
                                          llm_e0_core4_pruned_e0_only_h2_both4 or
                                          h2h7_both4 or h2_stable_leverage),
                             h2_restore=(llm_e0_core4_pruned_e0_only_h2_both4
                                        or e1_add_h2 or h2h7_both4
                                        or h2_stable_leverage),
                             h57_subscales=e2_h57_subscales,
                             h7_restore=h2h7_both4)
    if e6_add_noise_col:
        # ガードレール(-0.002)が本当に信号を検出しているか、完全な乱数列を
        # 1本足すだけでどれだけΔF1が動くかを測るための対照実験（2026-08-25）。
        X = X.copy()
        X["_NOISE_CONTROL"] = np.random.RandomState(0).uniform(size=len(X))
    txt = train[TEXT_COL].fillna("").astype(str).values
    org = load_org_text(train)
    # E4 の入力/モデル。org_embed=False なら現行と完全一致（e4=None 相当）。
    e4_in, _, e4_model = e4_inputs(train, train, org_embed, concat_embed)
    e4 = (e4_in, e4_model) if (org_embed or concat_embed) else None
    e7 = e7_model if cross else None
    lin = linear_c if linear else None
    h30_full = h30_inputs(h30_absdiff)
    h30 = ((h30_full[0], h30_full[2]) if h30_full is not None else None)
    gam_value = gam_c if gam else None
    knn = None
    if embed_knn:
        knn_train, _ = load_knn_embeddings()
        knn = (knn_train, build_knn_model)
    c0 = e0_anchor_cols(X.columns)
    if (llm_e0_core4_pruned or llm_e0_core4_pruned_e0_only_h2_both4
            or h2h7_both4 or h2_stable_leverage):
        missing = set(FINANCE_ISNA_COLS + SURVEY_ISNA_COLS)
        core4 = {"LLM_need_score", "LLM_make_limitation_score",
                 "LLM_buy_openness_score",
                 "LLM_internal_training_capability_score"}
        c0 = [c for c in c0 if not c.startswith("DX展望_") and c not in missing
              and (not c.startswith("LLM_") or c in core4)]
    c0_all = e0_anchor_cols(X.columns)
    c0b = ([c for c in c0_all if not c.startswith("LLM_")]
           if (llm_e0_core4_pruned_e0_only_h2_both4 or h2h7_both4
               or h2_stable_leverage) else c0)
    h2_mode = ("h2h7_both4" if h2h7_both4 else
               "h2_stable_leverage" if h2_stable_leverage else
               "h2_both4" if llm_e0_core4_pruned_e0_only_h2_both4 else None)
    if h2_mode is not None:
        c0, c0b = route_h2_anchor_cols(c0, c0b, h2_mode)
    c6 = e6_manual_cols(X.columns, drop_dx_manual=e6_drop_dx_manual,
                       add_llm4=e6_add_llm4, h52_prune=e6_h52_prune)
    if e6_add_noise_col:
        c6 = c6 + ["_NOISE_CONTROL"]
    cols = (c0, c0b,
            e1_finance_cols(with_ratios=finance_ratios, with_h2=e1_add_h2,
                            h54_prune=e1_h54_prune),
            e2_survey_cols(with_step=survey_step,
                          h57_subscales=e2_h57_subscales), c6,
            e7_cross_cols() if cross else None)
    print(f"A={int(len(y) * (1 - test_size))}行 / B(ホールドアウト)={int(len(y) * test_size)}行"
          f" x {n_reps}リピート, alpha={alpha}, メタ縮小先={target}"
          f"{', E1にH21財務比4本' if finance_ratios else ''}"
          f"{', E2にH22閾値指標4本' if survey_step else ''}"
          f"{', E4を[組織図;企業概要]連結embeddingに差し替え' if concat_embed else ''}"
          f"{', E4を組織図embeddingに差し替え' if org_embed and not concat_embed else ''}"
          f"{f', E7(H27 不満×財務, {e7_model})を追加' if cross else ''}"
          f"{f', E0b(H28 94列one-hot+LR, C={linear_c:g})を追加' if linear else ''}"
          f"{', E0/E0bをLLM親4軸のpruned構成へ変更' if llm_e0_core4_pruned else ''}"
          f"{', E0をpruned+LLM親4軸、H2財務4列をE0/E0bへ共有' if llm_e0_core4_pruned_e0_only_h2_both4 else ''}"
          f"{', E6のDX展望_手動辞書22列を削除' if e6_drop_dx_manual else ''}"
          f"{', E1にH2財務健全性比4本を追加' if e1_add_h2 else ''}"
          f"{', E6にH52 target-free重複/レアフラグ10本を削除' if e6_h52_prune else ''}"
          f"{', E1にH54 target-free会計恒等式重複5本を削除' if e1_h54_prune else ''}"
          f"{', E2をH57意味サブスケール4本へ置換(生11列は使わない)' if e2_h57_subscales else ''}"
          f"{', H2財務4本のうち負債比率/ROEをFCF比率/現金ROAへ差替(H2+H7)' if h2h7_both4 else ''}"
          f"{', H2の負債比率を安定版の負債資産比率へ差替' if h2_stable_leverage else ''}"
          f"{', E8(H30 3文書間絶対差embedding)を追加' if h30_absdiff else ''}"
          f"{f', E0c(H31 94列GAM, C={gam_c:g})を追加' if gam else ''}"
          f"{f', E9(H32 embedding cosine-kNN, dim={KNN_DIM}, k={KNN_NEIGHBORS}, scale={KNN_DISTANCE_SCALE:g}, prior={KNN_PRIOR_STRENGTH:g})を追加' if embed_knn else ''}"
          f"{', strict LGBM(Bラベル非参照)' if strict_lgbm else ''}")

    if seed_offset:
        print(f"seed_offset={seed_offset}: 未使用のfresh seedで監査 "
              f"(実seed={seed_offset}..{seed_offset + n_reps - 1})")
    recs, ws, alphas = [], [], []
    for rep in range(n_reps):
        seed = rep + seed_offset
        print(f"  rep {rep + 1}/{n_reps} (seed={seed}) ...", flush=True)
        r, w, a = one_rep(X, y, txt, org, cols, seed, alpha, test_size, rules,
                          target, e4, e7, lin, h30, gam_value, knn,
                          strict_lgbm)
        ws.append(w)
        alphas.append(a)
        for (m, rule), v in r.items():
            recs.append(dict(rep=rep, model=m, rule=rule, **v))
    if alpha is None:
        print(f"\nA内で選ばれたalpha (rep別): {alphas}")

    df = pd.DataFrame(recs)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        print(f"保存: {out_csv}")
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

    subs = subsets(cross, linear, h30_absdiff, gam, embed_knn)
    print("\n=== meta_6本(現本命) との対リピート差分 ===")
    for label in subs:
        if label == "meta_6本":
            continue
        for k in ["auc", "ap"]:
            d = piv[(k, label, ref)] - piv[(k, "meta_6本", ref)]
            print(f"  Δ{k.upper():3s} {label:8s}: mean {d.mean():+.4f}  "
                  f"正 {int((d > 0).sum())}/{len(d)}  各rep {np.round(d.values, 4)}")
        for r in rules:
            d = piv[("f1", label, r)] - piv[("f1", "meta_6本", r)]
            print(f"  ΔF1  {label:8s} {r:9s}: mean {d.mean():+.4f}  "
                  f"正 {int((d > 0).sum())}/{len(d)}  各rep {np.round(d.values, 4)}")

    if h30_absdiff:
        base_label = f"meta_{len(expert_names(cross, linear))}本"
        cand_label = f"meta_{len(expert_names(cross, linear, True))}本"
        base_df = df[(df.model == base_label) & (df.rule == ref)].sort_values("rep")
        cand_df = df[(df.model == cand_label) & (df.rule == ref)].sort_values("rep")
        result, deltas = verdict(base_df, cand_df)
        print(f"\n=== H30正式判定: {cand_label} - {base_label} ===")
        for key in ("auc", "ap", "f1"):
            d = deltas[key]
            print(f"  Δ{key.upper():3s}: mean {d['mean']:+.4f}  "
                  f"正 {d['n_pos']}/{d['n']}  min {d['min']:+.4f}  max {d['max']:+.4f}")
        print(f"  判定 => {result}")

    if gam:
        base_label = ("meta_8本" if cross and linear else
                      "meta_7本_lin" if linear else
                      "meta_7本" if cross else "meta_6本")
        cand_label = f"meta_{len(expert_names(cross, False, h30_absdiff, True))}本_gam"
        base_df = df[(df.model == base_label) & (df.rule == ref)].sort_values("rep")
        cand_df = df[(df.model == cand_label) & (df.rule == ref)].sort_values("rep")
        result, deltas = verdict(base_df, cand_df)
        print(f"\n=== H31正式判定（置換）: {cand_label} - {base_label} ===")
        for key in ("auc", "ap", "f1"):
            d = deltas[key]
            print(f"  Δ{key.upper():3s}: mean {d['mean']:+.4f}  "
                  f"正 {d['n_pos']}/{d['n']}  min {d['min']:+.4f}  max {d['max']:+.4f}")
        print(f"  判定 => {result}")

    if embed_knn:
        base_names = expert_names(cross, linear, h30_absdiff, gam)
        candidate_names = expert_names(cross, linear, h30_absdiff, gam, True)
        base_subsets = subsets(cross, linear, h30_absdiff, gam, False)
        base_label = next(
            label for label, members in reversed(list(base_subsets.items()))
            if tuple(members) == tuple(base_names)
        )
        candidate_label = f"meta_{len(candidate_names)}本_knn"
        base_df = df[(df.model == base_label) & (df.rule == ref)].sort_values("rep")
        candidate_df = df[(df.model == candidate_label) & (df.rule == ref)].sort_values("rep")
        result, deltas = verdict(base_df, candidate_df)
        print(f"\n=== H32正式判定: {candidate_label} - {base_label} ===")
        for key in ("auc", "ap", "f1"):
            d = deltas[key]
            print(f"  Δ{key.upper():3s}: mean {d['mean']:+.4f}  "
                  f"正 {d['n_pos']}/{d['n']}  min {d['min']:+.4f}  max {d['max']:+.4f}")
        print(f"  判定 => {result}")

    print("\n=== メタの重み (rep平均, 正規化%) ===")
    for label in subs:
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
    p.add_argument("--concat-embed", action="store_true",
                   help="E4 を [組織図 ; 企業概要] 連結 embedding+LR に差し替える"
                        "（--org-embed より優先。E0は不変）")
    p.add_argument("--cross", action="store_true",
                   help="H27 の E7(不満×財務)を7本目に足す。meta_7本 が候補になる")
    p.add_argument("--e7-model", default="lgbm", choices=["lgbm", "lr"],
                   help="E7 のモデル族（diagnose_cross.py で選ぶ）")
    p.add_argument("--linear", action="store_true",
                   help="H28 の E0b(94列 one-hot+LR)を足す")
    p.add_argument("--linear-c", type=float, default=LINEAR_C,
                   help="E0b の LR の C（既定 0.1）")
    p.add_argument("--h30-absdiff", action="store_true",
                   help="H30の3文書間絶対差embeddingを9本目に追加")
    p.add_argument("--gam", action="store_true",
                   help="H31の低自由度GAMをE0b置換候補として比較")
    p.add_argument("--gam-c", type=float, default=GAM_C,
                   help="H31 GAMのLR正則化C（事前固定値0.01）")
    p.add_argument("--embed-knn", action="store_true",
                   help="H32のDX展望+組織図embedding cosine-kNNを追加")
    p.add_argument("--llm-e0-core4-pruned", action="store_true",
                   help="E0/E0bのDX辞書・欠損flagsを削りLLM親4軸へ置換")
    p.add_argument("--llm-e0-core4-pruned-e0-only-h2-both4", action="store_true",
                   help="現行H2-both4: E0だけpruned+LLM親4軸、H2財務4列をE0/E0bへ共有")
    p.add_argument("--e6-drop-dx-manual", action="store_true",
                   help="E6からDX展望_手動辞書22列を削る(E0で実証済みのノイズ除去をE6にも適用)")
    p.add_argument("--e6-add-llm4", action="store_true",
                   help="--e6-drop-dx-manualと併用し、削った分をLLM親4軸で置換する")
    p.add_argument("--e1-add-h2", action="store_true",
                   help="E1にH2財務健全性比4本(自己資本比率/負債比率/ROA/ROE)を追加")
    p.add_argument("--e6-h52-prune", action="store_true",
                   help="E6のH52 target-free重複/レアフラグ10本を削除(signate_information_quality_research.md)")
    p.add_argument("--e1-h54-prune", action="store_true",
                   help="E1のH54 target-free会計恒等式重複5本を削除(signate_information_quality_research.md)")
    p.add_argument("--e6-add-noise-col", action="store_true",
                   help="ガードレール較正用: E6に完全な乱数列を1本追加するだけの対照実験")
    p.add_argument("--e2-h57-subscales", action="store_true",
                   help="E2を生11列からH57意味サブスケール4本へ置換(signate_information_quality_research.md)")
    p.add_argument("--h2h7-both4", action="store_true",
                   help="H2財務4本の負債比率/ROEをFCF比率/現金ROAへ差替(legacy_batch1で5/5満点だった組合せ)")
    p.add_argument("--h2-stable-leverage", action="store_true",
                   help="H2の負債比率(不安定, train std=588)を負債資産比率(std=0.22)へ差替")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="A/B分割・fold・モデルseedに一律で足すオフセット。"
                        "0..n_reps-1のdevelopment seedと重ならないfresh seedで"
                        "監査するときに使う(signate_information_quality_research.md 21節)")
    strict = p.add_mutually_exclusive_group()
    strict.add_argument("--strict-lgbm", dest="strict_lgbm", action="store_true",
                        help="LightGBMの木数を学習側だけで決定（既定）")
    strict.add_argument("--legacy-leaky-lgbm", dest="strict_lgbm",
                        action="store_false",
                        help="過去再現専用: 評価先ラベルをearly stoppingへ渡す旧方式。"
                             "本判定には使用禁止")
    p.set_defaults(strict_lgbm=True)
    p.add_argument("--out-csv", default=None,
                   help="repごとの全スコアをCSV保存")
    p.add_argument("--threshold-rules", nargs="+", default=["oof_best"],
                   choices=list(THRESHOLD_RULES),
                   help="比較する閾値則。先頭が差分の基準（既定は現行のoof_bestのみ）")
    p.add_argument("--meta-target", default="zero", choices=["zero", "uniform"],
                   help="メタ重みの縮小先。uniform は等分ブレンドへ縮む")
    a = p.parse_args()
    run(a.n_reps, a.alpha, a.test_size, a.finance_ratios,
        tuple(a.threshold_rules), a.meta_target, a.survey_step, a.org_embed,
        a.concat_embed, a.cross, a.e7_model, a.linear, a.linear_c,
        a.h30_absdiff, a.gam, a.gam_c, a.embed_knn, a.out_csv,
        a.llm_e0_core4_pruned, a.llm_e0_core4_pruned_e0_only_h2_both4,
        a.e6_drop_dx_manual, a.e6_add_llm4, a.e1_add_h2, a.e6_h52_prune,
        a.e1_h54_prune, a.e6_add_noise_col, a.seed_offset,
        a.e2_h57_subscales, a.h2h7_both4, a.h2_stable_leverage,
        a.strict_lgbm)
