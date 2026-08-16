"""Caruana greedy ensemble selection のprediction-level nested実装。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class GESConfig:
    max_iterations: int = 50
    f1_guard_tolerance: float = 0.0
    min_ap_gain_vs_base: float = 0.0


def _best_f1(y: np.ndarray, p: np.ndarray,
             thresholds: np.ndarray) -> tuple[float, float]:
    # sklearn.f1_scoreを閾値ごとに呼ぶとGES反復内で支配的になるため、同じ式を
    # (n_rows, n_thresholds) の配列演算でまとめて計算する。
    positive = y.astype(bool)[:, None]
    predicted = p[:, None] >= thresholds[None, :]
    tp = np.sum(predicted & positive, axis=0)
    fp = np.sum(predicted & ~positive, axis=0)
    fn = np.sum(~predicted & positive, axis=0)
    denominator = 2 * tp + fp + fn
    values = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                       where=denominator > 0)
    idx = int(np.argmax(values))
    return float(values[idx]), float(thresholds[idx])


def select_ensemble(y: np.ndarray, predictions: dict[str, np.ndarray],
                    base_name: str, thresholds: np.ndarray,
                    config: GESConfig):
    """train行だけで構成員とprefix長を決め、整数bag weightを返す。"""
    names = tuple(predictions)
    if base_name not in predictions:
        raise KeyError(f"base model not found: {base_name}")
    matrix = np.column_stack([predictions[name] for name in names])
    if not np.isfinite(matrix).all():
        raise ValueError("GES library contains non-finite predictions")

    base = predictions[base_name]
    base_ap = float(average_precision_score(y, base))
    base_f1, base_threshold = _best_f1(y, base, thresholds)

    running_sum = np.zeros(len(y), dtype=float)
    selected_indices: list[int] = []
    prefixes = []
    for iteration in range(1, config.max_iterations + 1):
        best_idx, best_ap = None, -np.inf
        for idx, _ in enumerate(names):
            candidate = (running_sum + matrix[:, idx]) / iteration
            ap = float(average_precision_score(y, candidate))
            # namesの順番を固定し、同値なら先に現れたモデルを選ぶ。
            if ap > best_ap + 1e-15:
                best_idx, best_ap = idx, ap
        assert best_idx is not None
        selected_indices.append(best_idx)
        running_sum += matrix[:, best_idx]
        blend = running_sum / iteration
        f1, threshold = _best_f1(y, blend, thresholds)
        prefixes.append({
            "iteration": iteration,
            "added": names[best_idx],
            "ap": best_ap,
            "f1": f1,
            "threshold": threshold,
        })

    eligible = [
        row for row in prefixes
        if row["f1"] >= base_f1 - config.f1_guard_tolerance
        and row["ap"] > base_ap + config.min_ap_gain_vs_base
    ]
    if eligible:
        # AP最大、同値なら短いprefixを採る。
        choice = max(eligible, key=lambda row: (row["ap"], -row["iteration"]))
        length = int(choice["iteration"])
        chosen = selected_indices[:length]
        fallback = False
    else:
        choice = {
            "iteration": 1, "added": base_name, "ap": base_ap,
            "f1": base_f1, "threshold": base_threshold,
        }
        chosen = [names.index(base_name)]
        fallback = True

    counts = Counter(names[idx] for idx in chosen)
    weights = {name: counts.get(name, 0) / len(chosen) for name in names}
    diagnostics = {
        "config": asdict(config),
        "base_ap": base_ap,
        "base_f1": base_f1,
        "base_threshold": base_threshold,
        "selected_ap": float(choice["ap"]),
        "selected_f1": float(choice["f1"]),
        "selected_threshold": float(choice["threshold"]),
        "selected_iterations": len(chosen),
        "fallback_to_base": fallback,
        "counts": dict(counts),
        "weights": weights,
        "search_trace": prefixes,
    }
    return weights, diagnostics


def apply_weights(predictions: dict[str, np.ndarray],
                  weights: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(next(iter(predictions.values()))), dtype=float)
    for name, weight in weights.items():
        out += weight * predictions[name]
    return out
