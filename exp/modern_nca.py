"""Structured-only ModernNCA の最小実装。

モデル構造と既定値は LAMDA-Tabular/TALENT の実装に合わせる。

  https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py
  https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/configs/default/modernNCA.json

TALENT 全体は多数の実験基盤依存を持つため、このリポジトリでは必要な
PLR embedding と ModernNCA 本体だけを小さく再実装する。予測は、学習した
Euclidean 空間にある fold-train label の softmax 加重平均である。
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModernNCAConfig:
    # TALENT/configs/default/modernNCA.json の model/training 既定値。
    dim: int = 128
    dropout: float = 0.1
    d_block: int = 512
    n_blocks: int = 0
    temperature: float = 1.0
    n_frequencies: int = 77
    frequency_scale: float = 0.04431360576139521
    d_embedding: int = 34
    sample_rate: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 0.0002

    # 今回の clean screening 用に事前固定した学習量。
    epochs: int = 80
    batch_size: int = 64
    threads: int = 4


class FoldPreprocessor:
    """fold-trainだけで median/standardization/OHE を学ぶ。"""

    def __init__(self, categorical_columns: tuple[str, ...]):
        self.categorical_columns = tuple(categorical_columns)

    def fit(self, X: pd.DataFrame):
        self.numeric_columns_ = [
            c for c in X.columns if c not in self.categorical_columns
        ]
        numeric = X[self.numeric_columns_].astype(float)
        self.medians_ = numeric.median().fillna(0.0)
        self.scaler_ = StandardScaler().fit(
            numeric.fillna(self.medians_).to_numpy(dtype=np.float64)
        )

        categorical = self._categorical_frame(X)
        self.onehot_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32,
        ).fit(categorical)
        return self

    def _categorical_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        # category dtype の fillna は新水準を直接入れられないため string 化する。
        return X[list(self.categorical_columns)].astype("string").fillna("__MISSING__")

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric = X[self.numeric_columns_].astype(float).fillna(self.medians_)
        x_num = self.scaler_.transform(
            numeric.to_numpy(dtype=np.float64)
        ).astype(np.float32)
        x_cat = self.onehot_.transform(self._categorical_frame(X)).astype(np.float32)
        if not np.isfinite(x_num).all() or not np.isfinite(x_cat).all():
            raise ValueError("ModernNCA preprocessing produced non-finite values")
        return x_num, x_cat


class PeriodicEmbeddings(nn.Module):
    """TALENT が使う trainable periodic numerical embedding。"""

    def __init__(self, n_features: int, n_frequencies: int,
                 frequency_scale: float):
        super().__init__()
        self.frequencies = nn.Parameter(torch.normal(
            0.0, frequency_scale, (n_features, n_frequencies),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * torch.pi * self.frequencies[None] * x[..., None]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class PLREmbeddings(nn.Sequential):
    """TALENT の ``PLREmbeddings(..., lite=True)`` と同じ構造。"""

    def __init__(self, n_features: int, n_frequencies: int,
                 frequency_scale: float, d_embedding: int):
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            nn.Linear(2 * n_frequencies, d_embedding),
            nn.ReLU(),
        )


class MLPBlock(nn.Module):
    def __init__(self, dim: int, d_block: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, d_block),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_block, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModernNCA(nn.Module):
    def __init__(self, n_num: int, n_cat: int, config: ModernNCAConfig):
        super().__init__()
        self.config = config
        self.num_embeddings = PLREmbeddings(
            n_num, config.n_frequencies, config.frequency_scale,
            config.d_embedding,
        )
        d_in = n_num * config.d_embedding + n_cat
        self.encoder = nn.Linear(d_in, config.dim)
        self.post_encoder = None
        if config.n_blocks > 0:
            self.post_encoder = nn.Sequential(
                *[MLPBlock(config.dim, config.d_block, config.dropout)
                  for _ in range(config.n_blocks)],
                nn.BatchNorm1d(config.dim),
            )

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.num_embeddings(x_num).flatten(1), x_cat], dim=1)
        x = self.encoder(x)
        if self.post_encoder is not None:
            x = self.post_encoder(x)
        return x

    def neighbor_log_proba(self, query_num: torch.Tensor,
                           query_cat: torch.Tensor,
                           candidate_num: torch.Tensor,
                           candidate_cat: torch.Tensor,
                           candidate_y: torch.Tensor,
                           diagonal_size: int = 0) -> torch.Tensor:
        query = self.encode(query_num, query_cat)
        candidate = self.encode(candidate_num, candidate_cat)
        distances = torch.cdist(query, candidate, p=2) / self.config.temperature
        if diagonal_size:
            # candidate の先頭は query と同じ順番。自分自身のlabelを除外する。
            diag = torch.arange(diagonal_size, device=distances.device)
            distances[diag, diag] = torch.inf
        weights = F.softmax(-distances, dim=1)
        onehot_y = F.one_hot(candidate_y, num_classes=2).to(weights.dtype)
        proba = weights @ onehot_y
        return torch.log(proba.clamp_min(1e-7))


def _set_deterministic(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True)


def _balanced_query_batches(y: torch.Tensor, batch_size: int,
                            generator: torch.Generator) -> list[torch.Tensor]:
    """各batchをほぼ1:1にし、query側の少数classを埋没させない。"""
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    half = min(batch_size // 2, len(pos), len(neg))
    if half < 1:
        raise ValueError("ModernNCA requires both classes in fold-train")
    n_batches = math.ceil(len(y) / (2 * half))
    batches = []
    for _ in range(n_batches):
        p = pos[torch.randperm(len(pos), generator=generator)[:half]]
        n = neg[torch.randperm(len(neg), generator=generator)[:half]]
        batch = torch.cat([p, n])
        batch = batch[torch.randperm(len(batch), generator=generator)]
        batches.append(batch)
    return batches


def _sample_candidates(n: int, query_idx: torch.Tensor, sample_rate: float,
                       generator: torch.Generator) -> torch.Tensor:
    all_idx = torch.arange(n)
    keep = torch.ones(n, dtype=torch.bool)
    keep[query_idx] = False
    rest = all_idx[keep]
    sample_size = max(1, int(len(rest) * sample_rate)) if len(rest) else 0
    if sample_size:
        sampled = rest[torch.randperm(len(rest), generator=generator)[:sample_size]]
        return torch.cat([query_idx, sampled])
    return query_idx


def fit_predict_fold(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_valid: pd.DataFrame, y_valid: np.ndarray | None,
                     categorical_columns: tuple[str, ...],
                     config: ModernNCAConfig, seed: int):
    """1 outer foldを学習し、outer-validをtrainだけの近傍DBで予測する。"""
    _set_deterministic(seed, config.threads)
    pre = FoldPreprocessor(categorical_columns).fit(X_train)
    tr_num, tr_cat = pre.transform(X_train)
    va_num, va_cat = pre.transform(X_valid)

    x_num = torch.from_numpy(tr_num)
    x_cat = torch.from_numpy(tr_cat)
    y = torch.as_tensor(np.asarray(y_train), dtype=torch.long)
    va_num_t = torch.from_numpy(va_num)
    va_cat_t = torch.from_numpy(va_cat)

    model = ModernNCA(x_num.shape[1], x_cat.shape[1], config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(config.epochs):
        model.train()
        epoch_loss = []
        for query_idx in _balanced_query_batches(y, config.batch_size, generator):
            candidate_idx = _sample_candidates(
                len(y), query_idx, config.sample_rate, generator,
            )
            log_proba = model.neighbor_log_proba(
                x_num[query_idx], x_cat[query_idx],
                x_num[candidate_idx], x_cat[candidate_idx], y[candidate_idx],
                diagonal_size=len(query_idx),
            )
            loss = F.nll_loss(log_proba, y[query_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))

    model.eval()
    with torch.no_grad():
        log_proba = model.neighbor_log_proba(
            va_num_t, va_cat_t, x_num, x_cat, y,
        )
        prediction = log_proba.exp()[:, 1].cpu().numpy()
        va_z = model.encode(va_num_t, va_cat_t)
        tr_z = model.encode(x_num, x_cat)
        nearest = torch.cdist(va_z, tr_z).argmin(dim=1)

    diagnostics = {
        "config": asdict(config),
        "n_num": int(x_num.shape[1]),
        "n_cat_onehot": int(x_cat.shape[1]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
    }
    if y_valid is not None:
        diagnostics["top1_same_label"] = float(
            (y[nearest].cpu().numpy() == y_valid).mean()
        )
    return prediction, diagnostics
