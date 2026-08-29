"""H68: soft industry-conditioned anti-template weighting for DX text.

H67 kept only the three most unusual sentences and lost too much of the raw
E3 signal.  H68 preserves every n-gram with at least ``BASE_WEIGHT`` and adds
the remaining mass according to each sentence's within-document novelty rank:

    sentence weight = 0.5 + 0.5 * relative_novelty

``relative_novelty`` is the percentile rank of the target-free cosine distance
from the shrunk industry centroid fitted by :class:`AntiTemplateSelector`.
The least novel sentence therefore keeps 50% of its count and the most novel
keeps 100%; nothing is deleted.  Weighted fractional counts then pass through
the same sublinear TF, training-fold IDF, L2 normalisation, and logistic
regression used by the current E3.

At ``base_weight=1.0`` this implementation is an exact numerical control for
the current E3.  This makes the H68 comparison isolate sentence weighting from
changes in vocabulary, IDF, classifier, or regularisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from anti_template_text_features import (
    GROUP_COL, AntiTemplateSelector, split_sentences,
)
from text_features import LR_PARAMS, TFIDF_PARAMS


SOFT_ANTI_TEMPLATE_NAME = "E13_soft_anti_template_text"
BASE_WEIGHT = 0.5


def relative_novelty(scores: np.ndarray) -> np.ndarray:
    """Map absolute cosine distances to stable within-document ranks [0, 1]."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=float)
    return (rankdata(scores, method="average") - 1.0) / (len(scores) - 1.0)


@dataclass(frozen=True)
class WeightDiagnostics:
    sentences: int
    mean_weight: float
    minimum_weight: float
    maximum_weight: float


class SoftAntiTemplateTransformer:
    """Create fold-fitted, novelty-weighted E3 sparse matrices."""

    def __init__(self, base_weight: float = BASE_WEIGHT):
        if not 0.0 <= base_weight <= 1.0:
            raise ValueError("base_weight must be between zero and one")
        self.base_weight = float(base_weight)

    def fit(self, texts, groups) -> "SoftAntiTemplateTransformer":
        texts = np.asarray(texts, dtype=object)
        groups = np.asarray(groups, dtype=object)
        if len(texts) != len(groups):
            raise ValueError("texts and groups must have the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty corpus")

        self.training_texts_ = texts
        self.training_groups_ = groups
        self.novelty_selector_ = AntiTemplateSelector().fit(texts, groups)
        self.training_novelty_ = (
            self.novelty_selector_.training_novelty_scores())

        count_params = {
            key: TFIDF_PARAMS[key]
            for key in ("analyzer", "ngram_range", "min_df")
        }
        self.count_vectorizer_ = CountVectorizer(**count_params)
        self.training_full_counts_ = self.count_vectorizer_.fit_transform(
            texts).tocsr()
        self.tfidf_transformer_ = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True,
            sublinear_tf=TFIDF_PARAMS["sublinear_tf"],
        ).fit(self.training_full_counts_)
        return self

    def _weighted_counts(self, texts, full_counts,
                         novelty_scores) -> sparse.csr_matrix:
        rows = []
        residual_scale = 1.0 - self.base_weight
        for doc_id, (text, scores) in enumerate(zip(texts, novelty_scores)):
            base = full_counts[doc_id].astype(float).multiply(self.base_weight)
            if residual_scale == 0.0:
                rows.append(base)
                continue

            sentences = split_sentences(text)
            if len(sentences) != len(scores):
                raise RuntimeError("sentence/novelty length mismatch")
            sentence_counts = self.count_vectorizer_.transform(
                sentences).tocsr()
            residual_weight = relative_novelty(scores) * residual_scale
            residual = (
                sparse.csr_matrix(residual_weight.reshape(1, -1))
                @ sentence_counts
            )
            rows.append((base + residual).tocsr())
        return sparse.vstack(rows, format="csr")

    def transform_training(self) -> sparse.csr_matrix:
        """Transform fitting documents using leave-one-company-out novelty."""
        weighted = self._weighted_counts(
            self.training_texts_, self.training_full_counts_,
            self.training_novelty_)
        return self.tfidf_transformer_.transform(weighted).tocsr()

    def transform(self, texts, groups) -> sparse.csr_matrix:
        """Transform documents excluded from all fitted text statistics."""
        texts = np.asarray(texts, dtype=object)
        groups = np.asarray(groups, dtype=object)
        if len(texts) != len(groups):
            raise ValueError("texts and groups must have the same length")
        novelty = self.novelty_selector_.novelty_scores(texts, groups)
        full_counts = self.count_vectorizer_.transform(texts).tocsr()
        weighted = self._weighted_counts(texts, full_counts, novelty)
        return self.tfidf_transformer_.transform(weighted).tocsr()

    def weight_diagnostics(self, novelty_scores=None) -> WeightDiagnostics:
        scores = (self.training_novelty_ if novelty_scores is None
                  else novelty_scores)
        weights = np.concatenate([
            self.base_weight
            + (1.0 - self.base_weight) * relative_novelty(doc_scores)
            for doc_scores in scores
        ])
        return WeightDiagnostics(
            sentences=len(weights),
            mean_weight=float(weights.mean()),
            minimum_weight=float(weights.min()),
            maximum_weight=float(weights.max()),
        )


def fit_predict_soft_anti_template(X, y, texts, train_indices,
                                   target_indices,
                                   base_weight: float = BASE_WEIGHT):
    """Fit H68 on ``train_indices`` and predict untouched target rows."""
    train_indices = np.asarray(train_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    texts = np.asarray(texts, dtype=object)
    groups = X[GROUP_COL].to_numpy(dtype=object)

    transformer = SoftAntiTemplateTransformer(base_weight=base_weight).fit(
        texts[train_indices], groups[train_indices])
    train_matrix = transformer.transform_training()
    target_matrix = transformer.transform(
        texts[target_indices], groups[target_indices])
    model = LogisticRegression(**LR_PARAMS).fit(
        train_matrix, np.asarray(y)[train_indices])
    probability = model.predict_proba(target_matrix)[:, 1]
    return probability, transformer, train_matrix, target_matrix
