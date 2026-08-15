"""R2: structured-only TabPFNをdiscovery内でexp032と比較する。

本文、TF-IDF、embedding、LLM score、手作業テキスト特徴は入れない。
``dataset.build_matrices`` の基本構造化列とH1利益率だけを使い、E0 LightGBM / E0b LRとは
異なるtabular priorがexp032の残差を説明するかを測る。

配管・model weight確認:

  OMP_NUM_THREADS=4 python3 exp/compare_tabpfn.py --smoke

Discovery 1-seed正式スクリーニング:

  OMP_NUM_THREADS=4 python3 exp/compare_tabpfn.py

外部pretrained weightを利用するため、大会画面上の現行ルールを確認できるまで、結果が良くても
lockbox・提出へは進めない。モデル・packageの利用許諾と大会ルールは別問題として扱う。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CAT_COLS, build_matrices  # noqa: E402
from ensemble_experts import _scores  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    INNER_FOLDS, LOCKBOX_SEED, _cross_fitted_meta, discovery_oof, fixed_split,
)


NAME = "E10_tabpfn"
TABPFN_VERSION = "8.3.0"
N_ESTIMATORS = 4
OUT_NPZ = Path("exp/_r2_tabpfn_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_r2_tabpfn_discovery_seed20260815.json")


def _candidate_model(seed: int, cat_indices: list[int], n_estimators: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=n_estimators,
        auto_scale_n_estimators=False,
        categorical_features_indices=cat_indices,
        device="cpu",
        fit_mode="fit_preprocessors",
        random_state=seed,
        n_preprocessing_jobs=4,
        show_progress_bar=False,
    )


def crossfit_candidate(X: pd.DataFrame, y: np.ndarray, seed: int,
                       n_estimators: int, smoke: bool = False):
    cat_indices = [X.columns.get_loc(c) for c in CAT_COLS]
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed,
    )
    oof = np.full(len(y), np.nan, dtype=float)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        if smoke and fold > 1:
            break
        print(f"TabPFN fold {fold}/{INNER_FOLDS}: train={len(tr)} valid={len(va)} "
              f"estimators={n_estimators}", flush=True)
        t0 = time.time()
        model = _candidate_model(seed + fold * 10000, cat_indices, n_estimators)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict_proba(X.iloc[va])[:, 1]
        seconds = time.time() - t0
        oof[va] = pred
        row = {
            "fold": fold,
            "seconds": seconds,
            "ap": float(average_precision_score(y[va], pred)),
            "auc": float(roc_auc_score(y[va], pred)),
        }
        diagnostics.append(row)
        print(f"  AP={row['ap']:.4f} AUC={row['auc']:.4f} "
              f"time={seconds:.0f}s", flush=True)
    return oof, diagnostics


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


def run(smoke: bool, refresh_base: bool):
    import tabpfn

    if tabpfn.__version__ != TABPFN_VERSION:
        raise RuntimeError(
            f"TabPFN version mismatch: expected {TABPFN_VERSION}, got {tabpfn.__version__}"
        )
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, _ = fixed_split(y_all)
    X_all, y_check, _ = build_matrices(train, train)
    assert np.array_equal(y_all, y_check)
    X = X_all.iloc[discovery].reset_index(drop=True)
    y = y_all[discovery]
    n_estimators = 1 if smoke else N_ESTIMATORS

    print(f"discovery={len(y)} positive_rate={y.mean():.4f} X={X.shape}")
    print(f"categorical={CAT_COLS} TabPFN={tabpfn.__version__} "
          f"n_estimators={n_estimators}")
    candidate_oof, diagnostics = crossfit_candidate(
        X, y, LOCKBOX_SEED, n_estimators, smoke=smoke,
    )
    if smoke:
        done = np.isfinite(candidate_oof)
        assert done.sum() > 0
        assert ((candidate_oof[done] >= 0) & (candidate_oof[done] <= 1)).all()
        print(f"\nSMOKE OK: {done.sum()}件をfold外予測。性能判定はしない。")
        return

    assert np.isfinite(candidate_oof).all(), "OOF未割当あり"
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
        "E0_anchor": _scores(y, base_oof["E0_anchor"]),
        "E0b_linear": _scores(y, base_oof["E0b_linear"]),
        NAME: _scores(y, candidate_oof),
        "exp032": _scores(y, base_blend),
        "exp032+tabpfn": _scores(y, candidate_blend),
    }
    table = pd.DataFrame(score_rows).T
    print("\n=== Discovery OOF ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    delta = (table.loc["exp032+tabpfn", ["auc", "ap", "f1"]]
             - table.loc["exp032", ["auc", "ap", "f1"]])
    corrs = {
        "E0_anchor": float(spearmanr(candidate_oof,
                                     base_oof["E0_anchor"]).statistic),
        "E0b_linear": float(spearmanr(candidate_oof,
                                      base_oof["E0b_linear"]).statistic),
        "exp032": float(spearmanr(candidate_oof, base_blend).statistic),
    }
    errors = _error_delta(
        y, base_blend, candidate_blend,
        table.loc["exp032", "th"], table.loc["exp032+tabpfn", "th"],
    )
    target_weight = float(weights[-1])

    print("\n=== exp032への増分 ===")
    print("  " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print(f"  TabPFN meta weight={target_weight:.4f}")
    print("  Spearman: " + " / ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    print("  Error delta: " + json.dumps(errors, ensure_ascii=False))
    print(f"  alpha per fold: {alphas}")

    result = {
        "tabpfn_version": tabpfn.__version__,
        "n_estimators": N_ESTIMATORS,
        "input_columns": list(X.columns),
        "categorical_columns": CAT_COLS,
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
        "competition_external_model_rule_verified": False,
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
    run(smoke=args.smoke, refresh_base=args.refresh_base)
