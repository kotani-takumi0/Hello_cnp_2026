"""H66: fold-safe counterfactual-twin preprocessing.

The current ensemble is strong on absolute company attributes but makes most
of its mistakes among otherwise ordinary, medium-sized companies.  This
module rewrites a company as five signed comparisons against two matched
reference groups:

    "How much closer is this company to matched purchasers than to matched
     non-purchasers in finance, survey, DX text, organisation text, and the
     company overview?"

Matching itself uses only stable context (industry, listing, business type,
employees, sales, and assets).  Outcome-bearing finance/survey/text fields do
not choose the twins.  For each class, the K context-nearest companies are
selected and a bounded distance margin is emitted for each of the five views.

The transform is supervised because reference labels determine the positive
and negative pools.  Leakage is prevented at two levels:

* ``CounterfactualTwinTransformer.transform`` never receives target labels.
* ``fit_predict_twin`` constructs the LR head's training features by an inner
  cross-fit.  A row's twin features therefore use neither that row nor its
  label.  The final target features use only the supplied training slice.

This file intentionally does not alter the production eight-expert pipeline.
It is consumed by ``compare_counterfactual_twin.py`` as an independent ninth
expert so the baseline predictions remain exactly paired.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from embedding_features import load_embeddings

TWIN_EXPERT_NAME = "E11_counterfactual_twin"
TWIN_NEIGHBORS = 5
TWIN_EMBED_DIM = 1024
TWIN_HEAD_C = 1.0

CONTEXT_CAT_COLS = ("業界", "上場種別", "特徴")
CONTEXT_NUM_COLS = ("従業員数", "売上", "総資産")
# Industry is the strongest peer definition; the other five coordinates have
# equal unit weight.  These constants are fixed before looking at H66 labels.
CONTEXT_WEIGHTS = np.asarray((2.0, 1.0, 1.0, 1.0, 1.0, 1.0))

FINANCE_VIEW_COLS = (
    "営業利益率",
    "純利益率",
    "自己資本比率",
    "負債資産比率",
    "ROA",
    "FCF比率",
    "現金ROA",
    "ソフト投資_売上比",
    "営業CFマージン",
    "有利子負債比率",
    "log_ソフト投資",
)
SURVEY_VIEW_COLS = tuple(f"アンケート{n}" for n in "１２３４５６７８９") + (
    "アンケート１０",
    "アンケート１１",
)
EMBEDDING_VIEWS = ("dx_outlook", "org", "overview")
FEATURE_NAMES = (
    "twin_margin_finance",
    "twin_margin_survey",
    "twin_margin_dx",
    "twin_margin_org",
    "twin_margin_overview",
)


def load_twin_embedding_views(dim: int = TWIN_EMBED_DIM):
    """Return aligned ``({train views}, {test views})`` embedding mappings."""
    train, test = {}, {}
    for slug in EMBEDDING_VIEWS:
        train[slug], test[slug] = load_embeddings(slug, dim=dim)
    return train, test


def _require_columns(frame: pd.DataFrame, columns, group: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{group}に必要な列が無い: {missing}")


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


@dataclass
class _NumericState:
    median: np.ndarray
    scale: np.ndarray


class CounterfactualTwinTransformer:
    """Fit peer matching and five-view distance margins on one reference set."""

    def __init__(self, n_neighbors: int = TWIN_NEIGHBORS):
        self.n_neighbors = int(n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")

    @staticmethod
    def _fit_numeric(frame: pd.DataFrame, columns) -> _NumericState:
        values = frame[list(columns)].astype(float).to_numpy()
        median = np.nanmedian(values, axis=0)
        # All-NaN is not expected, but a deterministic zero fallback makes a
        # fold with a rare missing column safe.
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(values), values, median)
        q25, q75 = np.quantile(filled, (0.25, 0.75), axis=0)
        scale = q75 - q25
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return _NumericState(median=median, scale=scale)

    @staticmethod
    def _transform_numeric(frame: pd.DataFrame, columns,
                           state: _NumericState) -> np.ndarray:
        values = frame[list(columns)].astype(float).to_numpy()
        values = np.where(np.isfinite(values), values, state.median)
        # asinh behaves linearly near the centre and logarithmically in the
        # tails.  That keeps distance from being dominated by one ratio while
        # retaining its sign (unlike clipping or abs-log).
        return np.arcsinh((values - state.median) / state.scale)

    @staticmethod
    def _fit_rank_reference(frame: pd.DataFrame):
        refs = []
        for column in CONTEXT_NUM_COLS:
            value = frame[column].astype(float).to_numpy()
            value = np.sign(value) * np.log1p(np.abs(value))
            finite = value[np.isfinite(value)]
            if not len(finite):
                finite = np.asarray([0.0])
            refs.append(np.sort(finite))
        return refs

    @staticmethod
    def _rank_context(frame: pd.DataFrame, rank_reference) -> np.ndarray:
        columns = []
        for column, reference in zip(CONTEXT_NUM_COLS, rank_reference):
            value = frame[column].astype(float).to_numpy()
            value = np.sign(value) * np.log1p(np.abs(value))
            fallback = float(np.median(reference))
            value = np.where(np.isfinite(value), value, fallback)
            columns.append(
                np.searchsorted(reference, value, side="right") / len(reference)
            )
        return np.column_stack(columns)

    @staticmethod
    def _categorical_context(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame[column].astype(object).where(frame[column].notna(),
                                                "__MISSING__").astype(str)
            for column in CONTEXT_CAT_COLS
        ])

    @staticmethod
    def _survey_values(frame: pd.DataFrame) -> pd.DataFrame:
        survey = frame[list(SURVEY_VIEW_COLS)].astype(float).copy()
        # Q7=0 is the pipeline's structural-missing marker (Q6=no).  Put it at
        # the established neutral value rather than treating it as extreme.
        survey["アンケート７"] = survey["アンケート７"].replace(0.0, 3.3)
        for column in SURVEY_VIEW_COLS:
            if column == "アンケート６":
                # Original coding: 1=yes, 2=no.  Both states get full range.
                survey[column] = 2.0 - survey[column]
            else:
                survey[column] = (survey[column] - 1.0) / 4.0
        return survey

    @classmethod
    def _survey_view(cls, frame: pd.DataFrame,
                     median: pd.Series) -> np.ndarray:
        return cls._survey_values(frame).fillna(median).to_numpy(dtype=float)

    def fit(self, frame: pd.DataFrame, y: np.ndarray,
            embeddings: dict[str, np.ndarray]):
        _require_columns(frame, CONTEXT_CAT_COLS + CONTEXT_NUM_COLS,
                         "peer context")
        _require_columns(frame, FINANCE_VIEW_COLS, "finance view")
        _require_columns(frame, SURVEY_VIEW_COLS, "survey view")
        missing_views = sorted(set(EMBEDDING_VIEWS) - set(embeddings))
        if missing_views:
            raise KeyError(f"embedding viewが無い: {missing_views}")

        self.y_ = np.asarray(y, dtype=int)
        if len(frame) != len(self.y_):
            raise ValueError("frame and y length mismatch")
        if set(np.unique(self.y_)) != {0, 1}:
            raise ValueError("reference data must contain both classes")

        self.rank_reference_ = self._fit_rank_reference(frame)
        self.context_num_ = self._rank_context(frame, self.rank_reference_)
        self.context_cat_ = self._categorical_context(frame)

        self.finance_state_ = self._fit_numeric(frame, FINANCE_VIEW_COLS)
        self.finance_ = self._transform_numeric(
            frame, FINANCE_VIEW_COLS, self.finance_state_)
        survey = self._survey_values(frame)
        self.survey_median_ = survey.median().fillna(0.5)
        self.survey_ = survey.fillna(self.survey_median_).to_numpy(dtype=float)
        self.embeddings_ = {}
        for name in EMBEDDING_VIEWS:
            matrix = np.asarray(embeddings[name], dtype=float)
            if len(matrix) != len(frame):
                raise ValueError(f"{name} embedding and frame length mismatch")
            self.embeddings_[name] = _normalise_rows(matrix)
        return self

    def _context_distance(self, frame: pd.DataFrame) -> np.ndarray:
        target_cat = self._categorical_context(frame)
        target_num = self._rank_context(frame, self.rank_reference_)
        parts = [
            (target_cat[:, i, None] != self.context_cat_[None, :, i]).astype(float)
            for i in range(len(CONTEXT_CAT_COLS))
        ]
        parts.extend([
            np.abs(target_num[:, i, None] - self.context_num_[None, :, i])
            for i in range(len(CONTEXT_NUM_COLS))
        ])
        distance = sum(
            weight * part for weight, part in zip(CONTEXT_WEIGHTS, parts)
        )
        return distance / CONTEXT_WEIGHTS.sum()

    def _nearest_by_class(self, context_distance: np.ndarray):
        indices = []
        for label in (1, 0):
            pool = np.flatnonzero(self.y_ == label)
            k = min(self.n_neighbors, len(pool))
            # Stable sort gives deterministic tie handling for quantised rows.
            order = np.argsort(context_distance[:, pool], axis=1,
                               kind="stable")[:, :k]
            indices.append(pool[order])
        return indices[0], indices[1]

    @staticmethod
    def _numeric_distance(target: np.ndarray, reference: np.ndarray,
                          index: np.ndarray) -> np.ndarray:
        delta = target[:, None, :] - reference[index]
        return np.sqrt(np.mean(delta * delta, axis=2)).mean(axis=1)

    @staticmethod
    def _cosine_distance(target: np.ndarray, reference: np.ndarray,
                         index: np.ndarray) -> np.ndarray:
        similarity = np.einsum("id,ikd->ik", target, reference[index])
        return (1.0 - np.clip(similarity, -1.0, 1.0)).mean(axis=1)

    @staticmethod
    def _margin(distance_positive: np.ndarray,
                distance_negative: np.ndarray) -> np.ndarray:
        # Positive means closer to purchaser twins.  Boundedness prevents one
        # anomalous document or accounting ratio from dominating the LR head.
        denominator = distance_positive + distance_negative
        return ((distance_negative - distance_positive)
                / np.maximum(denominator, 1e-8))

    def transform(self, frame: pd.DataFrame,
                  embeddings: dict[str, np.ndarray],
                  exclude_reference_diagonal: bool = False) -> np.ndarray:
        context_distance = self._context_distance(frame)
        if exclude_reference_diagonal:
            if len(frame) != len(self.y_):
                raise ValueError("diagonal exclusion requires reference-sized input")
            np.fill_diagonal(context_distance, np.inf)
        positive, negative = self._nearest_by_class(context_distance)

        finance = self._transform_numeric(
            frame, FINANCE_VIEW_COLS, self.finance_state_)
        survey = self._survey_view(frame, self.survey_median_)
        output = []
        for target, reference in (
            (finance, self.finance_),
            (survey, self.survey_),
        ):
            d_pos = self._numeric_distance(target, reference, positive)
            d_neg = self._numeric_distance(target, reference, negative)
            output.append(self._margin(d_pos, d_neg))

        for name in EMBEDDING_VIEWS:
            target = _normalise_rows(np.asarray(embeddings[name], dtype=float))
            if len(target) != len(frame):
                raise ValueError(f"{name} embedding and frame length mismatch")
            reference = self.embeddings_[name]
            d_pos = self._cosine_distance(target, reference, positive)
            d_neg = self._cosine_distance(target, reference, negative)
            output.append(self._margin(d_pos, d_neg))

        matrix = np.column_stack(output)
        if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
            raise RuntimeError("counterfactual twin transform produced invalid values")
        return matrix


def _slice_embeddings(embeddings: dict[str, np.ndarray], index) -> dict:
    return {name: np.asarray(matrix)[index] for name, matrix in embeddings.items()}


def fit_predict_twin(X: pd.DataFrame, y: np.ndarray,
                     embeddings: dict[str, np.ndarray], train_index,
                     target_index, seed: int,
                     n_neighbors: int = TWIN_NEIGHBORS,
                     head_c: float = TWIN_HEAD_C) -> np.ndarray:
    """Fit the nested twin expert on ``train_index`` and predict target rows.

    ``y[target_index]`` is deliberately never indexed.  Training features for
    the five-column head are themselves cross-fitted inside ``train_index``.
    """
    train_index = np.asarray(train_index, dtype=int)
    target_index = np.asarray(target_index, dtype=int)
    y_train = np.asarray(y, dtype=int)[train_index]
    X_train = X.iloc[train_index]

    twin_oof = np.zeros((len(train_index), len(FEATURE_NAMES)), dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for reference_local, valid_local in folds.split(X_train, y_train):
        transformer = CounterfactualTwinTransformer(n_neighbors).fit(
            X_train.iloc[reference_local], y_train[reference_local],
            _slice_embeddings(embeddings, train_index[reference_local]),
        )
        twin_oof[valid_local] = transformer.transform(
            X_train.iloc[valid_local],
            _slice_embeddings(embeddings, train_index[valid_local]),
        )

    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=head_c, max_iter=3000, random_state=0),
    )
    head.fit(twin_oof, y_train)

    transformer = CounterfactualTwinTransformer(n_neighbors).fit(
        X_train, y_train, _slice_embeddings(embeddings, train_index)
    )
    twin_target = transformer.transform(
        X.iloc[target_index], _slice_embeddings(embeddings, target_index)
    )
    return head.predict_proba(twin_target)[:, 1]
