"""R4: structured-only ModernNCAをdiscovery内でexp032と比較する。

  OMP_NUM_THREADS=4 python3 exp/compare_modern_nca.py --smoke
  OMP_NUM_THREADS=4 python3 exp/compare_modern_nca.py

入力は ``dataset.build_matrices`` の基本構造化47列だけ。各outer validation行は
学習時にも近傍DBにも入れず、lockbox 260件は一切評価しない。正式スクリーニングの
epochは80に固定し、結果を見た追加探索はしない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CAT_COLS, build_matrices  # noqa: E402
from embedding_knn import build_knn_model, load_knn_embeddings  # noqa: E402
from ensemble_experts import _scores  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    INNER_FOLDS, LOCKBOX_SEED, _cross_fitted_meta, discovery_oof, fixed_split,
)
from modern_nca import ModernNCAConfig, fit_predict_fold  # noqa: E402


NAME = "E10_modern_nca"
OUT_NPZ = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_r4_modern_nca_discovery_seed20260815.json")


def crossfit_candidates(X: pd.DataFrame, y: np.ndarray,
                        fixed_knn_x: np.ndarray,
                        config: ModernNCAConfig, seed: int,
                        smoke: bool = False):
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed,
    )
    nca_oof = np.full(len(y), np.nan, dtype=float)
    knn_oof = np.full(len(y), np.nan, dtype=float)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        if smoke and fold > 1:
            break
        print(f"ModernNCA fold {fold}/{INNER_FOLDS}: "
              f"train={len(tr)} valid={len(va)} epochs={config.epochs}", flush=True)
        t0 = time.time()
        pred, diag = fit_predict_fold(
            X.iloc[tr], y[tr], X.iloc[va], y[va], tuple(CAT_COLS), config,
            seed=seed + fold * 10000,
        )
        nca_oof[va] = pred

        # exp035 fixed cosine-kNNも全く同じ discovery/fold で比較する。
        knn = build_knn_model().fit(fixed_knn_x[tr], y[tr])
        knn_oof[va] = knn.predict_proba(fixed_knn_x[va])[:, 1]

        diag.update({
            "fold": fold,
            "seconds": time.time() - t0,
            "ap": float(average_precision_score(y[va], pred)),
            "auc": float(roc_auc_score(y[va], pred)),
            "fixed_knn_ap": float(average_precision_score(y[va], knn_oof[va])),
        })
        diagnostics.append(diag)
        print(f"  AP={diag['ap']:.4f} AUC={diag['auc']:.4f} "
              f"fixed-kNN AP={diag['fixed_knn_ap']:.4f} "
              f"loss={diag['initial_loss']:.4f}->{diag['final_loss']:.4f} "
              f"time={diag['seconds']:.0f}s", flush=True)
    return nca_oof, knn_oof, diagnostics


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


def run(config: ModernNCAConfig, smoke: bool, refresh_base: bool):
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, _ = fixed_split(y_all)
    X_all, y_check, _ = build_matrices(train, train)
    assert np.array_equal(y_all, y_check)
    X = X_all.iloc[discovery].reset_index(drop=True)
    y = y_all[discovery]

    knn_all, _ = load_knn_embeddings()
    fixed_knn_x = knn_all[discovery]
    effective = replace(config, epochs=3) if smoke else config
    print(f"discovery={len(y)} positive_rate={y.mean():.4f} X={X.shape}")
    print("config=" + json.dumps(asdict(effective), ensure_ascii=False))
    nca_oof, knn_oof, diagnostics = crossfit_candidates(
        X, y, fixed_knn_x, effective, LOCKBOX_SEED, smoke=smoke,
    )
    if smoke:
        done = np.isfinite(nca_oof)
        assert done.sum() > 0
        assert ((nca_oof[done] >= 0) & (nca_oof[done] <= 1)).all()
        print(f"\nSMOKE OK: {done.sum()}件をfold外予測。性能判定はしない。")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return

    assert np.isfinite(nca_oof).all() and np.isfinite(knn_oof).all()
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

    candidate_experts = {**base_oof, NAME: nca_oof}
    candidate_names = names + (NAME,)
    candidate_blend, alphas, weights = _cross_fitted_meta(
        y, candidate_experts, candidate_names, LOCKBOX_SEED,
    )
    score_rows = {
        "exp035_fixed_knn": _scores(y, knn_oof),
        "E0_anchor": _scores(y, base_oof["E0_anchor"]),
        "E0b_linear": _scores(y, base_oof["E0b_linear"]),
        NAME: _scores(y, nca_oof),
        "exp032": _scores(y, base_blend),
        "exp032+modern_nca": _scores(y, candidate_blend),
    }
    table = pd.DataFrame(score_rows).T
    print("\n=== Discovery OOF ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    delta = (table.loc["exp032+modern_nca", ["auc", "ap", "f1"]]
             - table.loc["exp032", ["auc", "ap", "f1"]])
    corrs = {
        "exp035_fixed_knn": float(spearmanr(nca_oof, knn_oof).statistic),
        "E0_anchor": float(spearmanr(nca_oof, base_oof["E0_anchor"]).statistic),
        "E0b_linear": float(spearmanr(nca_oof, base_oof["E0b_linear"]).statistic),
        "exp032": float(spearmanr(nca_oof, base_blend).statistic),
    }
    errors = _error_delta(
        y, base_blend, candidate_blend,
        table.loc["exp032", "th"], table.loc["exp032+modern_nca", "th"],
    )
    target_weight = float(weights[-1])
    print("\n=== exp032への増分 ===")
    print("  " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print(f"  ModernNCA meta weight={target_weight:.4f}")
    print("  Spearman: " + " / ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    print("  Error delta: " + json.dumps(errors, ensure_ascii=False))
    print(f"  alpha per fold: {alphas}")

    result = {
        "implementation_reference": (
            "https://github.com/LAMDA-Tabular/TALENT/tree/main/TALENT/model"
        ),
        "config": asdict(config),
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "scores": {k: {m: float(v) for m, v in row.items()}
                   for k, row in score_rows.items()},
        "delta_vs_exp032": {k: float(v) for k, v in delta.items()},
        "spearman": corrs,
        "errors": errors,
        "target_weight": target_weight,
        "alphas": [float(v) for v in alphas],
        "fold_diagnostics": diagnostics,
        "lockbox_opened": False,
    }
    np.savez_compressed(
        OUT_NPZ, y=y, modern_nca_oof=nca_oof, fixed_knn_oof=knn_oof,
        base_blend=base_blend, candidate_blend=candidate_blend,
    )
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"保存: {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refresh-base", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(
        ModernNCAConfig(epochs=args.epochs, threads=args.threads),
        smoke=args.smoke,
        refresh_base=args.refresh_base,
    )
