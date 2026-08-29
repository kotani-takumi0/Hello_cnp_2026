"""H67: industry-conditioned anti-template preprocessing for DX outlook text.

The raw ``今後のDX展望`` field contains roughly twelve sentences per company.
Instead of asking a supervised model which sentence matters, this transformer
estimates boilerplate without using ``購入フラグ``:

1. fit a character TF-IDF space on sentences from the fitting fold only;
2. build an industry centroid, shrunk toward the global centroid;
3. rank each document's sentences by distance from that centroid; and
4. retain the three most company-specific sentences in their original order.

When transforming the fitting documents, the current document is removed from
both centroids.  Thus a sentence never makes itself look more conventional and
the same out-of-sample scoring rule is used for fitting and target documents.
The selected text is passed to the unchanged E3 TF-IDF + logistic-regression
classifier from :mod:`text_features`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from text_features import build_model


ANTI_TEMPLATE_NAME = "E12_anti_template_text"
GROUP_COL = "業界"
TOP_SENTENCES = 3
SHRINKAGE_SENTENCES = 100.0
MIN_SENTENCE_CHARS = 12

SELECTOR_TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(2, 4),
    min_df=3,
    sublinear_tf=True,
    norm="l2",
)

_BOUNDARY = re.compile(r"(?<=[。！？!?])|[\r\n]+")
_SPACE = re.compile(r"[\s　]+")


def _group_key(value: object) -> str:
    """Return a stable key for category values, including missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "__MISSING__"
    return str(value)


def split_sentences(text: object) -> list[str]:
    """Split text conservatively and discard headings/tiny fragments."""
    raw = str(text) if text is not None else ""
    pieces = [_SPACE.sub(" ", part).strip() for part in _BOUNDARY.split(raw)]
    sentences = [part for part in pieces if len(part) >= MIN_SENTENCE_CHARS]
    if sentences:
        return sentences
    fallback = _SPACE.sub(" ", raw).strip()
    return [fallback] if fallback else ["記載なし"]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


@dataclass(frozen=True)
class SelectionDiagnostics:
    documents: int
    sentences_before: int
    sentences_after: int
    characters_before: int
    characters_after: int

    @property
    def character_retention(self) -> float:
        if self.characters_before == 0:
            return 0.0
        return self.characters_after / self.characters_before


class AntiTemplateSelector:
    """Target-free, fold-fitted selector of company-specific sentences."""

    def __init__(self, top_sentences: int = TOP_SENTENCES,
                 shrinkage_sentences: float = SHRINKAGE_SENTENCES):
        if top_sentences < 1:
            raise ValueError("top_sentences must be positive")
        if shrinkage_sentences < 0:
            raise ValueError("shrinkage_sentences must be non-negative")
        self.top_sentences = int(top_sentences)
        self.shrinkage_sentences = float(shrinkage_sentences)

    def fit(self, texts, groups) -> "AntiTemplateSelector":
        texts = np.asarray(texts, dtype=object)
        groups = np.asarray(groups, dtype=object)
        if len(texts) != len(groups):
            raise ValueError("texts and groups must have the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty corpus")

        docs = [split_sentences(text) for text in texts]
        flat = [sentence for doc in docs for sentence in doc]
        spans = []
        cursor = 0
        for doc in docs:
            spans.append((cursor, cursor + len(doc)))
            cursor += len(doc)

        self.vectorizer_ = TfidfVectorizer(**SELECTOR_TFIDF_PARAMS)
        matrix = self.vectorizer_.fit_transform(flat).tocsr()
        self.training_sentences_ = docs
        self.training_groups_ = np.array([_group_key(g) for g in groups])
        self.training_spans_ = spans
        self.training_matrix_ = matrix
        self.global_sum_ = np.asarray(matrix.sum(axis=0)).ravel()
        self.global_count_ = matrix.shape[0]

        self.group_sums_: dict[str, np.ndarray] = {}
        self.group_counts_: dict[str, int] = {}
        for key in np.unique(self.training_groups_):
            doc_ids = np.flatnonzero(self.training_groups_ == key)
            rows = np.concatenate([
                np.arange(spans[i][0], spans[i][1]) for i in doc_ids
            ])
            self.group_sums_[key] = np.asarray(matrix[rows].sum(axis=0)).ravel()
            self.group_counts_[key] = int(len(rows))

        self.global_centroid_ = _unit(self.global_sum_ / self.global_count_)
        self.group_centroids_ = {
            key: self._centroid(key) for key in self.group_sums_
        }
        return self

    def _centroid(self, key: str, excluded_sum: np.ndarray | None = None,
                  excluded_count: int = 0) -> np.ndarray:
        """Return the shrunk group centroid, optionally leaving one doc out."""
        if excluded_sum is None:
            excluded_sum = np.zeros_like(self.global_sum_)
        global_count = self.global_count_ - excluded_count
        if global_count <= 0:
            return self.global_centroid_
        global_mean = (self.global_sum_ - excluded_sum) / global_count

        group_sum = self.group_sums_.get(key)
        group_count = self.group_counts_.get(key, 0)
        if group_sum is None:
            return _unit(global_mean)
        group_count -= excluded_count
        group_sum = group_sum - excluded_sum
        if group_count <= 0:
            return _unit(global_mean)

        numerator = group_sum + self.shrinkage_sentences * global_mean
        denominator = group_count + self.shrinkage_sentences
        return _unit(numerator / denominator)

    def _select(self, sentences: list[str], matrix: sparse.csr_matrix,
                centroid: np.ndarray) -> str:
        similarities = np.asarray(matrix @ centroid).ravel()
        novelty = 1.0 - similarities
        keep = min(self.top_sentences, len(sentences))
        # mergesort makes ties deterministic; sort again to preserve narrative order.
        selected = np.argsort(-novelty, kind="mergesort")[:keep]
        selected.sort()
        return " ".join(sentences[i] for i in selected)

    @staticmethod
    def _novelty(matrix: sparse.csr_matrix,
                 centroid: np.ndarray) -> np.ndarray:
        """Return bounded cosine distance for every sentence row."""
        similarities = np.asarray(matrix @ centroid).ravel()
        return np.clip(1.0 - similarities, 0.0, 1.0)

    def training_novelty_scores(self) -> list[np.ndarray]:
        """Score fitted documents with document-level leave-one-out centroids."""
        output = []
        for doc_id, (start, stop) in enumerate(self.training_spans_):
            matrix = self.training_matrix_[start:stop]
            doc_sum = np.asarray(matrix.sum(axis=0)).ravel()
            centroid = self._centroid(
                self.training_groups_[doc_id], doc_sum, stop - start)
            output.append(self._novelty(matrix, centroid))
        return output

    def novelty_scores(self, texts, groups) -> list[np.ndarray]:
        """Score documents that did not contribute to the fitted centroids."""
        texts = np.asarray(texts, dtype=object)
        groups = np.asarray(groups, dtype=object)
        if len(texts) != len(groups):
            raise ValueError("texts and groups must have the same length")
        output = []
        for text, group in zip(texts, groups):
            sentences = split_sentences(text)
            matrix = self.vectorizer_.transform(sentences).tocsr()
            centroid = self.group_centroids_.get(
                _group_key(group), self.global_centroid_)
            output.append(self._novelty(matrix, centroid))
        return output

    def transform_training(self) -> np.ndarray:
        """Transform the fitted documents with document-level leave-one-out."""
        output = []
        for doc_id, (start, stop) in enumerate(self.training_spans_):
            matrix = self.training_matrix_[start:stop]
            doc_sum = np.asarray(matrix.sum(axis=0)).ravel()
            centroid = self._centroid(
                self.training_groups_[doc_id], doc_sum, stop - start)
            output.append(self._select(
                self.training_sentences_[doc_id], matrix, centroid))
        return np.asarray(output, dtype=object)

    def transform(self, texts, groups) -> np.ndarray:
        """Transform documents that did not contribute to fitted centroids."""
        texts = np.asarray(texts, dtype=object)
        groups = np.asarray(groups, dtype=object)
        if len(texts) != len(groups):
            raise ValueError("texts and groups must have the same length")
        output = []
        for text, group in zip(texts, groups):
            sentences = split_sentences(text)
            matrix = self.vectorizer_.transform(sentences).tocsr()
            centroid = self.group_centroids_.get(
                _group_key(group), self.global_centroid_)
            output.append(self._select(sentences, matrix, centroid))
        return np.asarray(output, dtype=object)

    @staticmethod
    def diagnostics(raw_texts, selected_texts) -> SelectionDiagnostics:
        raw = np.asarray(raw_texts, dtype=object)
        selected = np.asarray(selected_texts, dtype=object)
        return SelectionDiagnostics(
            documents=len(raw),
            sentences_before=sum(len(split_sentences(text)) for text in raw),
            sentences_after=sum(len(split_sentences(text)) for text in selected),
            characters_before=sum(len(str(text)) for text in raw),
            characters_after=sum(len(str(text)) for text in selected),
        )


def fit_predict_anti_template(X, y, texts, train_indices, target_indices):
    """Fit H67 on ``train_indices`` and predict ``target_indices``.

    ``y[target_indices]`` is never read.  Selector fitting is also independent
    of every label, so flipping any labels while keeping training labels fixed
    must leave the returned probabilities unchanged.
    """
    train_indices = np.asarray(train_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    groups = X[GROUP_COL].to_numpy(dtype=object)
    selector = AntiTemplateSelector().fit(
        np.asarray(texts, dtype=object)[train_indices], groups[train_indices])
    selected_train = selector.transform_training()
    selected_target = selector.transform(
        np.asarray(texts, dtype=object)[target_indices], groups[target_indices])
    model = build_model().fit(selected_train, np.asarray(y)[train_indices])
    probabilities = model.predict_proba(selected_target)[:, 1]
    return probabilities, selector, selected_train, selected_target
