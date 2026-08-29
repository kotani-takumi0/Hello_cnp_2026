"""H69: a nested, signed residual expert from H67 and the current E3.

The non-negative meta learner can add the hard anti-template expert, but it
cannot express ``hard - beta * full`` when the useful part requires a negative
coefficient on the full-text E3.  H69 makes that contrast explicit:

1. obtain cross-fitted full-text and H67 probabilities on the fitting rows;
2. regress ``logit(H67)`` on ``logit(E3)`` without labels;
3. retain the orthogonal signed residual; and
4. fit a one-variable logistic calibrator on fitting labels.

For every target fold, step 1 is itself cross-fitted inside the reference
rows.  Target labels and target text statistics are never read during fitting.
The output is therefore a regular probability expert that the existing
non-negative meta learner can safely accept or set to zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anti_template_text_features import fit_predict_anti_template
from meta_blend import to_logit
from text_features import build_model


CONTRASTIVE_RESIDUAL_NAME = "E14_contrastive_text_residual"
INNER_SPLITS = 5
INNER_SEED = 0
CALIBRATOR_C = 1.0


@dataclass(frozen=True)
class ResidualDiagnostics:
    projection_intercept: float
    projection_slope: float
    residual_mean: float
    residual_std: float
    residual_full_correlation: float
    calibrator_coefficient: float
    calibrator_intercept: float


def fit_apply_residual(full_reference_oof, hard_reference_oof, y_reference,
                       full_target, hard_target):
    """Fit the target-free projection and supervised 1-D calibrator."""
    full_reference_logit = to_logit(full_reference_oof)
    hard_reference_logit = to_logit(hard_reference_oof)
    design = np.column_stack([
        np.ones(len(full_reference_logit)), full_reference_logit
    ])
    projection_intercept, projection_slope = np.linalg.lstsq(
        design, hard_reference_logit, rcond=None)[0]
    residual = hard_reference_logit - (
        projection_intercept + projection_slope * full_reference_logit)

    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=CALIBRATOR_C, max_iter=2000, random_state=0),
    )
    calibrator.fit(residual.reshape(-1, 1), np.asarray(y_reference))

    target_residual = to_logit(hard_target) - (
        projection_intercept + projection_slope * to_logit(full_target))
    target_probability = calibrator.predict_proba(
        target_residual.reshape(-1, 1))[:, 1]
    reference_probability = calibrator.predict_proba(
        residual.reshape(-1, 1))[:, 1]

    if residual.std() > 0.0 and full_reference_logit.std() > 0.0:
        correlation = float(np.corrcoef(
            residual, full_reference_logit)[0, 1])
    else:
        correlation = 0.0
    lr = calibrator.named_steps["logisticregression"]
    diagnostics = ResidualDiagnostics(
        projection_intercept=float(projection_intercept),
        projection_slope=float(projection_slope),
        residual_mean=float(residual.mean()),
        residual_std=float(residual.std()),
        residual_full_correlation=correlation,
        calibrator_coefficient=float(lr.coef_[0, 0]),
        calibrator_intercept=float(lr.intercept_[0]),
    )
    return target_probability, diagnostics, reference_probability


def fit_predict_contrastive_residual(
        X, y, texts, reference_indices, target_indices,
        target_full_probability=None, target_hard_probability=None,
        inner_seed: int = INNER_SEED):
    """Predict target rows from a nested cross-fitted H67-minus-E3 residual."""
    reference_indices = np.asarray(reference_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y)

    full_oof = np.zeros(len(reference_indices))
    hard_oof = np.zeros(len(reference_indices))
    folds = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=inner_seed)
    for inner_train, inner_valid in folds.split(
            reference_indices, y[reference_indices]):
        train_rows = reference_indices[inner_train]
        valid_rows = reference_indices[inner_valid]
        full_model = build_model().fit(texts[train_rows], y[train_rows])
        full_oof[inner_valid] = full_model.predict_proba(
            texts[valid_rows])[:, 1]
        hard_oof[inner_valid] = fit_predict_anti_template(
            X, y, texts, train_rows, valid_rows)[0]

    if ((target_full_probability is None)
            != (target_hard_probability is None)):
        raise ValueError("target full/hard probabilities must be supplied together")
    if target_full_probability is None:
        full_model = build_model().fit(
            texts[reference_indices], y[reference_indices])
        target_full_probability = full_model.predict_proba(
            texts[target_indices])[:, 1]
        target_hard_probability = fit_predict_anti_template(
            X, y, texts, reference_indices, target_indices)[0]
    else:
        target_full_probability = np.asarray(target_full_probability)
        target_hard_probability = np.asarray(target_hard_probability)
        if (len(target_full_probability) != len(target_indices)
                or len(target_hard_probability) != len(target_indices)):
            raise ValueError("precomputed target probability length mismatch")

    target_probability, diagnostics, reference_probability = (
        fit_apply_residual(
            full_oof, hard_oof, y[reference_indices],
            target_full_probability, target_hard_probability)
    )
    return (
        target_probability,
        diagnostics,
        full_oof,
        hard_oof,
        reference_probability,
    )
