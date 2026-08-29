"""H65: unsupervised archetype expert (E10) -- soft GMM segmentation over
financial-health ratios + survey items, feeding a low-freedom linear head.

Motivation (`documents/deep-research-report.md` 最終診断 / user request for a
"completely different angle" even at the cost of standalone score): every
remaining expert either reuses the tree's own axis-aligned split geometry
(E1 raw amounts, E0/E6 LightGBM) or a linear boundary in the *same* raw
column space (E0b). H52/H54/H56 all show that compressing or pruning columns
*inside* an existing expert's input does not survive to the final blend --
LightGBM already reconstructs whatever a dedup/PCA could offer via splits.
H5's industry target-encoding was REJECTed as actively harmful because that
geometry (categorical group means) is exactly what LightGBM's native
categorical split already extracts.

This tries a geometry none of E0/E1/E2/E6/E7/E0b can produce: soft cluster
membership in a *continuous multi-column* space that mixes two axes no
existing expert combines directly -- financial-health ratios (H1/H2/H7,
scale-invariant) and survey attitude (E2's raw columns) -- fit by a
generative density model (GMM) instead of a discriminative split/linear
model. Two safeguards against repeating H5's failure mode:

  1. The clustering itself never sees ``y`` (unlike target encoding) -- it is
     fit purely on standardized ratios/survey values.
  2. The prediction head is a plain L2 logistic regression over the K soft
     membership probabilities (K<=6), not a raw per-cluster target rate --
     this avoids the small-cluster overfitting a hard mean-encoding would
     hit at n~480/fold.

Fold-safety: the median imputer, StandardScaler, and GaussianMixture are all
fit on the training slice only; the target slice only ever gets
``.transform()`` / ``.predict_proba()``. Component count K is chosen by BIC
computed on the training slice alone (still target-free, no y involved).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# H1 (会計利益率) + H2/H7 (財務健全性・現金ベース投資余力) の生成済み比率7列。
# 規模で正規化済みなので、E1(生金額)やE0/E0bと同じスケール軸を再エンコード
# しない。build_features(h2_restore=True, h7_restore=True) 済みのXに存在する。
FINANCE_HEALTH_COLS = (
    "営業利益率", "経常利益率", "純利益率",
    "自己資本比率", "ROA", "FCF比率", "現金ROA",
)
SURVEY_RAW_COLS = tuple(f"アンケート{n}" for n in "１２３４５６７８９") + (
    "アンケート１０", "アンケート１１",
)
ARCHETYPE_COLS = FINANCE_HEALTH_COLS + SURVEY_RAW_COLS

N_COMPONENTS_GRID = (3, 4, 5, 6)
HEAD_C = 1.0
GMM_N_INIT = 5
GMM_REG_COVAR = 1e-3


def _prepare(frame, median):
    return frame[list(ARCHETYPE_COLS)].astype(float).fillna(median)


def _select_n_components(Z, seed):
    """BIC最小のK。yは一切参照しない（target-free）。"""
    best_k, best_bic = N_COMPONENTS_GRID[0], np.inf
    for k in N_COMPONENTS_GRID:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=seed, n_init=2,
                              reg_covar=GMM_REG_COVAR)
        gmm.fit(Z)
        bic = gmm.bic(Z)
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def fit_predict_archetype(Xtr, ytr, Xtarget, seed, n_components=None):
    """Xtr で学習し、Xtarget の購入確率を返す。他expertの `_strict_lgbm_predict`
    と同じ (Xtr, ytr, Xtarget, seed) -> np.ndarray の形にしてある。
    """
    med = Xtr[list(ARCHETYPE_COLS)].astype(float).median()
    raw_tr, raw_ta = _prepare(Xtr, med), _prepare(Xtarget, med)

    scaler = StandardScaler().fit(raw_tr)
    Ztr, Zta = scaler.transform(raw_tr), scaler.transform(raw_ta)

    k = n_components or _select_n_components(Ztr, seed)
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          random_state=seed, n_init=GMM_N_INIT,
                          reg_covar=GMM_REG_COVAR)
    gmm.fit(Ztr)

    Rtr, Rta = gmm.predict_proba(Ztr), gmm.predict_proba(Zta)
    head = LogisticRegression(C=HEAD_C, max_iter=2000, random_state=0)
    head.fit(Rtr, ytr)
    return head.predict_proba(Rta)[:, 1]
