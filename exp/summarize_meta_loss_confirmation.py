"""Paired uncertainty summary for strict meta-loss holdout experiments."""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"logloss", "brier", "raw_logloss", "raw_brier"}


def exact_sign_flip_p_better(effect: np.ndarray) -> float:
    """Exact one-sided paired randomization p-value under sign symmetry."""
    effect = np.asarray(effect, dtype=float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(effect))))
    null_mean = (signs * np.abs(effect)).mean(axis=1)
    return float(np.mean(null_mean >= effect.mean() - 1e-15))


def summarize(frame: pd.DataFrame, baseline: str, metrics: list[str],
              bootstrap_reps: int, seed: int) -> pd.DataFrame:
    pivot = frame.pivot(index="seed", columns="loss")
    losses = [name for name in frame["loss"].unique() if name != baseline]
    rng = np.random.default_rng(seed)
    rows = []
    for candidate in losses:
        for metric in metrics:
            delta = (pivot[(metric, candidate)]
                     - pivot[(metric, baseline)]).to_numpy(dtype=float)
            benefit = -delta if metric in LOWER_IS_BETTER else delta
            sampled_index = rng.integers(
                0, len(delta), size=(bootstrap_reps, len(delta)))
            bootstrap_delta = delta[sampled_index].mean(axis=1)
            low, high = np.quantile(bootstrap_delta, (0.025, 0.975))
            rows.append({
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "n_pairs": len(delta),
                "baseline_mean": float(pivot[(metric, baseline)].mean()),
                "candidate_mean": float(pivot[(metric, candidate)].mean()),
                "delta_candidate_minus_baseline": float(delta.mean()),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "wins": int((benefit > 1e-12).sum()),
                "ties": int((np.abs(benefit) <= 1e-12).sum()),
                "losses": int((benefit < -1e-12).sum()),
                "one_sided_p_better": exact_sign_flip_p_better(benefit),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--baseline", default="logloss")
    parser.add_argument("--metrics", nargs="+", default=[
        "f1", "ap", "logloss", "brier", "mcc", "outer_oracle_f1",
    ])
    parser.add_argument("--bootstrap-reps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = summarize(
        pd.read_csv(arguments.input), arguments.baseline, arguments.metrics,
        arguments.bootstrap_reps, arguments.seed)
    result.to_csv(arguments.output, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"saved: {arguments.output}")

