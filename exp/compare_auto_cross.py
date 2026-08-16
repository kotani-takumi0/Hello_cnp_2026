"""AutoCross-inspired 2-way cross + sparse LR のDiscovery評価。

  OMP_NUM_THREADS=4 python3 exp/compare_auto_cross.py --smoke
  OMP_NUM_THREADS=4 python3 exp/compare_auto_cross.py

outer foldごとにinner CVを作り、quantile binとcross選択をouter-train内に閉じる。
lockbox 260件は読み出しも評価もしない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_cross import AutoCrossConfig, fit_predict_outer  # noqa: E402
from ensemble_experts import _scores  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    INNER_FOLDS, LOCKBOX_SEED, _cross_fitted_meta, discovery_oof, fixed_split,
)
from meta_blend import apply_meta, fit_meta, to_logit  # noqa: E402


NAME = "E10_auto_cross"
OUT_NPZ = Path("exp/_autocross_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_autocross_discovery_seed20260815.json")
FIXED_ALPHAS = (0.001, 0.003, 0.01, 0.03)


def crossfit_candidate(raw: pd.DataFrame, y: np.ndarray,
                       config: AutoCrossConfig, seed: int,
                       smoke: bool = False):
    folds = StratifiedKFold(
        INNER_FOLDS, shuffle=True, random_state=seed,
    )
    oof = np.full(len(y), np.nan, dtype=float)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(raw, y), 1):
        if smoke and fold > 1:
            break
        print(f"AutoCross fold {fold}/{INNER_FOLDS}: "
              f"train={len(tr)} valid={len(va)}", flush=True)
        t0 = time.time()
        pred, diag = fit_predict_outer(
            raw.iloc[tr], y[tr], raw.iloc[va], config,
            seed=seed + fold * 10000,
            candidate_limit=24 if smoke else None,
        )
        oof[va] = pred
        diag.update({
            "fold": fold,
            "seconds": time.time() - t0,
            "ap": float(average_precision_score(y[va], pred)),
            "auc": float(roc_auc_score(y[va], pred)),
        })
        diagnostics.append(diag)
        print(f"  selected={len(diag['selected_crosses'])} "
              f"inner ΔAP={diag['selected_inner_gain']:+.4f} "
              f"outer AP={diag['ap']:.4f} AUC={diag['auc']:.4f} "
              f"time={diag['seconds']:.0f}s", flush=True)
        if diag["selected_crosses"]:
            print("  " + " / ".join(diag["selected_crosses"]), flush=True)
    return oof, diagnostics


def _fixed_meta(y, oof, names, seed, alpha):
    z = to_logit(np.column_stack([oof[n] for n in names]))
    out = np.zeros(len(y), dtype=float)
    weights = []
    folds = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    for tr, va in folds.split(z, y):
        w, b = fit_meta(z[tr], y[tr], alpha=alpha, target="zero")
        out[va] = apply_meta(z[va], w, b)
        weights.append(w)
    return out, np.mean(weights, axis=0)


def _stability(diagnostics):
    sets = [set(d["selected_crosses"]) for d in diagnostics]
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    counts = Counter(cross for s in sets for cross in s)
    return {
        "mean_pairwise_jaccard": float(np.mean(values)),
        "selected_count_per_fold": [len(s) for s in sets],
        "selection_frequency": dict(sorted(
            counts.items(), key=lambda row: (-row[1], row[0])
        )),
    }


def _error_delta(y, base, cand, base_th, cand_th):
    base_label = base >= base_th
    cand_label = cand >= cand_th
    return {
        "base_errors": int((base_label != y).sum()),
        "candidate_errors": int((cand_label != y).sum()),
        "fn_rescued": int(((y == 1) & ~base_label & cand_label).sum()),
        "new_fn": int(((y == 1) & base_label & ~cand_label).sum()),
        "fp_removed": int(((y == 0) & base_label & ~cand_label).sum()),
        "new_fp": int(((y == 0) & ~base_label & cand_label).sum()),
        "label_disagreement": int((base_label != cand_label).sum()),
    }


def run(config: AutoCrossConfig, smoke: bool, refresh_base: bool):
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, _ = fixed_split(y_all)
    raw = train.iloc[discovery].reset_index(drop=True)
    y = y_all[discovery]
    effective = replace(config, screen_size=8, max_crosses=3,
                        inner_folds=2) if smoke else config
    print(f"discovery={len(y)} positive_rate={y.mean():.4f}")
    print("config=" + json.dumps(asdict(effective), ensure_ascii=False))
    candidate_oof, diagnostics = crossfit_candidate(
        raw, y, effective, LOCKBOX_SEED, smoke=smoke,
    )
    if smoke:
        done = np.isfinite(candidate_oof)
        assert done.sum() > 0
        assert ((candidate_oof[done] >= 0) & (candidate_oof[done] <= 1)).all()
        print(f"\nSMOKE OK: {done.sum()}件をnested fold外予測。性能判定はしない。")
        return

    assert np.isfinite(candidate_oof).all()
    cache = Path("exp/_r1_exp032_discovery.npz")
    if cache.exists() and not refresh_base:
        z = np.load(cache, allow_pickle=True)
        names = tuple(z["names"].tolist())
        base_oof = {n: z[f"oof_{n}"] for n in names}
        base_blend = z["blend"]
        assert np.array_equal(z["y"], y)
        print(f"exp032 discovery cache: {cache}")
    else:
        print("exp032 discovery OOFを生成 ...", flush=True)
        y_base, base_oof, base_blend, names, _, _ = discovery_oof(
            train, discovery, LOCKBOX_SEED,
        )
        assert np.array_equal(y_base, y)
        np.savez_compressed(
            cache, y=y, names=np.array(names), blend=base_blend,
            **{f"oof_{n}": base_oof[n] for n in names},
        )

    candidate_experts = {**base_oof, NAME: candidate_oof}
    candidate_names = names + (NAME,)
    candidate_blend, alphas, weights = _cross_fitted_meta(
        y, candidate_experts, candidate_names, LOCKBOX_SEED,
    )
    score_rows = {
        "E7_cross": _scores(y, base_oof["E7_cross"]),
        "E0b_linear": _scores(y, base_oof["E0b_linear"]),
        NAME: _scores(y, candidate_oof),
        "exp032": _scores(y, base_blend),
        "exp032+auto_cross": _scores(y, candidate_blend),
    }
    table = pd.DataFrame(score_rows).T
    print("\n=== Discovery OOF ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    delta = (table.loc["exp032+auto_cross", ["auc", "ap", "f1"]]
             - table.loc["exp032", ["auc", "ap", "f1"]])

    fixed_rows = []
    for alpha in FIXED_ALPHAS:
        pb, _ = _fixed_meta(y, base_oof, names, LOCKBOX_SEED, alpha)
        pc, wc = _fixed_meta(
            y, candidate_experts, candidate_names, LOCKBOX_SEED, alpha,
        )
        sb, sc = _scores(y, pb), _scores(y, pc)
        fixed_rows.append({
            "alpha": alpha,
            "delta_auc": sc["auc"] - sb["auc"],
            "delta_ap": sc["ap"] - sb["ap"],
            "delta_f1": sc["f1"] - sb["f1"],
            "candidate_weight": float(wc[-1]),
        })
    fixed = pd.DataFrame(fixed_rows)

    corrs = {
        "E7_cross": float(spearmanr(candidate_oof, base_oof["E7_cross"]).statistic),
        "E0b_linear": float(spearmanr(candidate_oof, base_oof["E0b_linear"]).statistic),
        "exp032": float(spearmanr(candidate_oof, base_blend).statistic),
    }
    stability = _stability(diagnostics)
    errors = _error_delta(
        y, base_blend, candidate_blend,
        table.loc["exp032", "th"], table.loc["exp032+auto_cross", "th"],
    )
    print("\n=== exp032への増分 ===")
    print("  " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print(f"  AutoCross meta weight={float(weights[-1]):.4f}")
    print("  Spearman: " + " / ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    print(f"  selected Jaccard={stability['mean_pairwise_jaccard']:.3f}")
    print("\n=== 固定alpha感度 ===")
    print(fixed.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("  Error delta: " + json.dumps(errors, ensure_ascii=False))

    result = {
        "config": asdict(config),
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "scores": {k: {m: float(v) for m, v in row.items()}
                   for k, row in score_rows.items()},
        "delta_vs_exp032": {k: float(v) for k, v in delta.items()},
        "fixed_alpha": fixed_rows,
        "spearman": corrs,
        "stability": stability,
        "errors": errors,
        "target_weight": float(weights[-1]),
        "alphas": [float(v) for v in alphas],
        "fold_diagnostics": diagnostics,
        "lockbox_opened": False,
    }
    np.savez_compressed(
        OUT_NPZ, y=y, candidate_oof=candidate_oof,
        base_blend=base_blend, candidate_blend=candidate_blend,
    )
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"保存: {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh-base", action="store_true")
    args = parser.parse_args()
    run(AutoCrossConfig(), smoke=args.smoke, refresh_base=args.refresh_base)
