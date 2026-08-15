"""Discovery用 AutoCross-inspired 2-way cross探索。

AutoCrossそのものの探索器を再現するのではなく、今回のロードマップで固定した
最小構成を実装する。

* category / survey ordinal / fold内quantile-binを原子にする
* 2-way crossをinner CVのAPだけでscreening + greedy選択する
* 最終予測器はone-hot sparse Logistic Regression

bin境界、one-hot水準、cross選択はすべて呼び出し側のouter-train内で完結する。
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from cross_features import RATIO_COLS, add_cross_features


CATEGORICAL_ATOMS = ("業界", "上場種別", "特徴")
SURVEY_ATOMS = tuple(f"アンケート{i}" for i in
                     ("１", "２", "３", "４", "５", "６",
                      "７", "８", "９", "１０", "１１"))
NUMERIC_ATOMS = (
    "従業員数", "売上", "総資産", "資本金",
    "営業利益率_AC", "経常利益率_AC", "純利益率_AC",
) + tuple(RATIO_COLS)
ATOM_NAMES = CATEGORICAL_ATOMS + SURVEY_ATOMS + NUMERIC_ATOMS
ALL_PAIRS = tuple(itertools.combinations(ATOM_NAMES, 2))


@dataclass(frozen=True)
class AutoCrossConfig:
    n_bins: int = 5
    inner_folds: int = 3
    lr_c: float = 0.1
    screen_size: int = 40
    max_crosses: int = 20
    min_ap_gain: float = 0.001
    max_iter: int = 2000


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.astype(float) / b.astype(float)).replace([np.inf, -np.inf], np.nan)


def _numeric_frame(raw: pd.DataFrame) -> pd.DataFrame:
    derived = add_cross_features(raw, pd.DataFrame(index=raw.index))
    out = pd.DataFrame(index=raw.index)
    for col in ("従業員数", "売上", "総資産", "資本金"):
        out[col] = raw[col].astype(float)
    out["営業利益率_AC"] = _safe_div(raw["営業利益"], raw["売上"])
    out["経常利益率_AC"] = _safe_div(raw["経常利益"], raw["売上"])
    out["純利益率_AC"] = _safe_div(raw["当期純利益"], raw["売上"])
    for col in RATIO_COLS:
        out[col] = derived[col].astype(float)
    return out[list(NUMERIC_ATOMS)]


class AtomTransformer:
    """数値だけfold内quantile-bin化し、すべてstring categoryへ変換する。"""

    def __init__(self, n_bins: int):
        self.n_bins = int(n_bins)

    def fit(self, raw: pd.DataFrame):
        numeric = _numeric_frame(raw)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = {}
        for col in NUMERIC_ATOMS:
            values = numeric[col].dropna().to_numpy(dtype=float)
            edges = np.quantile(values, qs) if len(values) else np.array([])
            self.edges_[col] = np.unique(edges[np.isfinite(edges)])
        return self

    @staticmethod
    def _strings(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__MISSING__").astype(str)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        for col in CATEGORICAL_ATOMS + SURVEY_ATOMS:
            out[col] = self._strings(raw[col])
        numeric = _numeric_frame(raw)
        for col in NUMERIC_ATOMS:
            values = numeric[col].to_numpy(dtype=float)
            label = np.full(len(values), "__MISSING__", dtype=object)
            ok = np.isfinite(values)
            label[ok] = np.char.add(
                "b", np.digitize(values[ok], self.edges_[col]).astype(str),
            )
            out[col] = label
        return out[list(ATOM_NAMES)]


def cross_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__X__{pair[1]}"


def add_cross_columns(atoms: pd.DataFrame,
                      pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]]
                      ) -> pd.DataFrame:
    out = atoms.copy()
    for left, right in pairs:
        out[cross_name((left, right))] = (
            atoms[left].astype(str) + "||" + atoms[right].astype(str)
        )
    return out


def _model(config: AutoCrossConfig, seed: int):
    return LogisticRegression(
        C=config.lr_c, solver="liblinear", max_iter=config.max_iter,
        random_state=seed,
    )


def _encoded_fold(raw_train: pd.DataFrame, raw_valid: pd.DataFrame,
                  all_pairs: tuple[tuple[str, str], ...],
                  config: AutoCrossConfig):
    transformer = AtomTransformer(config.n_bins).fit(raw_train)
    train_atoms = transformer.transform(raw_train)
    valid_atoms = transformer.transform(raw_valid)
    train_frame = add_cross_columns(train_atoms, all_pairs)
    valid_frame = add_cross_columns(valid_atoms, all_pairs)
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float64,
    )
    train_matrix = encoder.fit_transform(train_frame)
    valid_matrix = encoder.transform(valid_frame)

    slices = {}
    offset = 0
    for col, categories in zip(train_frame.columns, encoder.categories_):
        width = len(categories)
        slices[col] = np.arange(offset, offset + width, dtype=int)
        offset += width
    base_indices = np.concatenate([slices[c] for c in ATOM_NAMES])
    pair_indices = {pair: slices[cross_name(pair)] for pair in all_pairs}
    return train_matrix.tocsr(), valid_matrix.tocsr(), base_indices, pair_indices


def _selection_cache(raw: pd.DataFrame, y: np.ndarray,
                     all_pairs: tuple[tuple[str, str], ...],
                     config: AutoCrossConfig, seed: int):
    folds = StratifiedKFold(
        config.inner_folds, shuffle=True, random_state=seed,
    )
    cache = []
    for tr, va in folds.split(raw, y):
        xtr, xva, base_idx, pair_idx = _encoded_fold(
            raw.iloc[tr], raw.iloc[va], all_pairs, config,
        )
        cache.append({
            "xtr": xtr, "xva": xva, "ytr": y[tr], "yva": y[va],
            "va": va, "base_idx": base_idx, "pair_idx": pair_idx,
        })
    return cache


def _indices(item, selected: tuple[tuple[str, str], ...] | list[tuple[str, str]]):
    if not selected:
        return item["base_idx"]
    return np.concatenate(
        [item["base_idx"]] + [item["pair_idx"][pair] for pair in selected]
    )


def _cv_ap(cache, y: np.ndarray,
           selected: tuple[tuple[str, str], ...] | list[tuple[str, str]],
           config: AutoCrossConfig, seed: int) -> float:
    oof = np.zeros(len(y), dtype=float)
    for fold, item in enumerate(cache):
        idx = _indices(item, selected)
        model = _model(config, seed + fold)
        model.fit(item["xtr"][:, idx], item["ytr"])
        oof[item["va"]] = model.predict_proba(item["xva"][:, idx])[:, 1]
    return float(average_precision_score(y, oof))


def select_crosses(raw: pd.DataFrame, y: np.ndarray,
                   config: AutoCrossConfig, seed: int,
                   candidate_limit: int | None = None):
    all_pairs = ALL_PAIRS if candidate_limit is None else ALL_PAIRS[:candidate_limit]
    cache = _selection_cache(raw, y, all_pairs, config, seed)
    base_ap = _cv_ap(cache, y, (), config, seed)

    screened = []
    for pair in all_pairs:
        ap = _cv_ap(cache, y, (pair,), config, seed)
        screened.append((pair, ap, ap - base_ap))
    screened.sort(key=lambda row: (-row[1], cross_name(row[0])))
    shortlist = [row[0] for row in screened[:config.screen_size]]

    selected: list[tuple[str, str]] = []
    trace = []
    current_ap = base_ap
    remaining = shortlist.copy()
    for _ in range(config.max_crosses):
        trials = []
        for pair in remaining:
            ap = _cv_ap(cache, y, selected + [pair], config, seed)
            trials.append((pair, ap))
        if not trials:
            break
        pair, best_ap = max(trials, key=lambda row: (row[1], cross_name(row[0])))
        gain = best_ap - current_ap
        if gain < config.min_ap_gain:
            break
        selected.append(pair)
        remaining.remove(pair)
        trace.append({
            "cross": cross_name(pair), "inner_ap": best_ap,
            "step_gain": gain,
        })
        current_ap = best_ap

    diagnostics = {
        "config": asdict(config),
        "n_candidates": len(all_pairs),
        "base_inner_ap": base_ap,
        "selected_inner_ap": current_ap,
        "selected_inner_gain": current_ap - base_ap,
        "selected_crosses": [cross_name(pair) for pair in selected],
        "selection_trace": trace,
        "screen_top": [
            {"cross": cross_name(pair), "ap": ap, "delta": delta}
            for pair, ap, delta in screened[:config.screen_size]
        ],
    }
    return tuple(selected), diagnostics


def fit_predict_outer(raw_train: pd.DataFrame, y_train: np.ndarray,
                      raw_valid: pd.DataFrame, config: AutoCrossConfig,
                      seed: int, candidate_limit: int | None = None):
    selected, diagnostics = select_crosses(
        raw_train, y_train, config, seed, candidate_limit=candidate_limit,
    )
    transformer = AtomTransformer(config.n_bins).fit(raw_train)
    tr = add_cross_columns(transformer.transform(raw_train), selected)
    va = add_cross_columns(transformer.transform(raw_valid), selected)
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float64,
    )
    xtr = encoder.fit_transform(tr)
    xva = encoder.transform(va)
    model = _model(config, seed).fit(xtr, y_train)
    pred = model.predict_proba(xva)[:, 1]
    diagnostics["encoded_features"] = int(xtr.shape[1])
    return pred, diagnostics

