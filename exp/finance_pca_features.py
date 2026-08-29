"""H56: fold-safe block PCA compression for E1's correlated finance columns.

`documents/signate_information_quality_research.md` (section 24) ranked block
PCA as the fourth-priority information-quality experiment for E1, after H52
(E6 dedup, REJECTed neutral), H53 (skipped), and H54 (E1 target-free
redundancy pruning by dropping near-duplicate raw columns, also REJECTed
neutral -- see documents/h52_h60_and_h2_leverage_20260826.md). H54's Spearman
analysis on signed-log-transformed FINANCE_COLS (train, y never touched)
found two tight blocks:

  size:   総資産 / 流動資産 / 固定資産 / 負債 / 純資産 / 自己資本 (rho >= 0.991)
  profit: 営業利益 / 経常利益 / 当期純利益 / 営業CF        (rho >= 0.993)

H54 handled that redundancy by *deleting* 5 of the 10 columns down to a
single "most complete" representative per near-duplicate pair -- a lossy
operation that discards whatever unique variance the dropped columns still
carried. PCA (Hotelling 1933) instead projects the whole block onto its
leading components, retaining a weighted combination of all columns' signal
(the "shared size factor" as PC1, remaining divergence as PC2) rather than
picking a winner. This is the literature-standard alternative to H54's
heuristic pruning for exactly this problem shape (a cluster of near-collinear
raw measurements), and was never actually run.

PCA is fit per outer-training-fold only (target-free, no leakage) on the
signed-log + standardized block, exactly mirroring the fold-safe pattern
already used for TF-IDF and org-text embeddings elsewhere in this project.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

FINANCE_SIZE_BLOCK = ("総資産", "流動資産", "固定資産", "負債", "純資産", "自己資本")
FINANCE_PROFIT_BLOCK = ("営業利益", "経常利益", "当期純利益", "営業CF")


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fold_finance_pca(
        Xtr: pd.DataFrame, Xva: pd.DataFrame, block: tuple[str, ...],
        prefix: str, n_components: int,
        seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fit signed-log + standardize + PCA on the train fold only; transform both."""
    columns = list(block)
    train_block = Xtr[columns].astype(float)
    valid_block = Xva[columns].astype(float)
    medians = train_block.median()
    train_block = train_block.fillna(medians)
    valid_block = valid_block.fillna(medians)

    train_log = _signed_log(train_block.to_numpy())
    valid_log = _signed_log(valid_block.to_numpy())
    mean = train_log.mean(axis=0)
    std = train_log.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_z = (train_log - mean) / std
    valid_z = (valid_log - mean) / std

    pca = PCA(n_components=n_components, random_state=seed)
    with warnings.catch_warnings():
        # Apple Accelerate BLAS emits a spurious "divide by zero encountered
        # in matmul" RuntimeWarning on some finite, correct matmuls; verified
        # numerically (no NaN/Inf in outputs, explained_variance_ratio_ sums
        # to ~1) that this is a benign backend false positive, not a real
        # numerical problem with the near-collinear input blocks.
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in matmul")
        warnings.filterwarnings(
            "ignore", message="overflow encountered in matmul")
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in matmul")
        train_scores = pca.fit_transform(train_z)
        valid_scores = pca.transform(valid_z)
    score_cols = [f"{prefix}{i + 1}" for i in range(n_components)]
    train_frame = pd.DataFrame(train_scores, columns=score_cols, index=Xtr.index)
    valid_frame = pd.DataFrame(valid_scores, columns=score_cols, index=Xva.index)
    return train_frame, valid_frame, pca.explained_variance_ratio_


def e1_pca_frames(
        Xtr: pd.DataFrame, Xva: pd.DataFrame, other_cols: list[str],
        seed: int = 0, size_components: int = 2,
        profit_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the size/profit blocks with PCA scores; keep every other E1 column."""
    size_tr, size_va, _ = fold_finance_pca(
        Xtr, Xva, FINANCE_SIZE_BLOCK, "財務PCA_size", size_components, seed)
    profit_tr, profit_va, _ = fold_finance_pca(
        Xtr, Xva, FINANCE_PROFIT_BLOCK, "財務PCA_profit", profit_components, seed)
    train_frame = pd.concat(
        [Xtr[other_cols].reset_index(drop=True),
         size_tr.reset_index(drop=True), profit_tr.reset_index(drop=True)],
        axis=1)
    valid_frame = pd.concat(
        [Xva[other_cols].reset_index(drop=True),
         size_va.reset_index(drop=True), profit_va.reset_index(drop=True)],
        axis=1)
    train_frame.index = Xtr.index
    valid_frame.index = Xva.index
    return train_frame, valid_frame
