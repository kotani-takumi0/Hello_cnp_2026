"""R6: 同一Discovery OOFライブラリ上のNested Caruana GES。

  python3 exp/compare_nested_ges.py --smoke
  python3 exp/compare_nested_ges.py

各outer foldのtrain行だけで構成員と反復数を選び、outer-validationへ固定weightを
適用する。APをgreedy objective、exp032 train-F1以上をguardrailとする。

H33 multi-seed OOFは742件CVでlockbox labelを学習側に含むため、Discovery protocolへ
混ぜない。R2 TabPFNはweight未取得なので候補なし。利用するのは固定Discovery上で
生成済みのclean OOFだけである。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS, _scores  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    INNER_FOLDS, LOCKBOX_SEED, _cross_fitted_meta,
)
from nested_ges import GESConfig, apply_weights, select_ensemble  # noqa: E402


BASE_CACHE = Path("exp/_r1_exp032_discovery.npz")
R1_CACHE = Path("exp/_r1_target_text_discovery_seed20260815.npz")
R4_CACHE = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
AUTOCROSS_CACHE = Path("exp/_autocross_discovery_seed20260815.npz")
OUT_NPZ = Path("exp/_r6_nested_ges_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_r6_nested_ges_discovery_seed20260815.json")
BASE_NAME = "exp032"


def _load_vector(path: Path, key: str, y: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"必要なDiscovery cacheがない: {path}")
    data = np.load(path, allow_pickle=True)
    if not np.array_equal(data["y"], y):
        raise RuntimeError(f"Discovery行順が一致しない: {path}")
    value = np.asarray(data[key], dtype=float)
    if value.shape != y.shape or not np.isfinite(value).all():
        raise RuntimeError(f"不正な予測配列: {path}:{key}")
    return value


def load_library(seed: int):
    if not BASE_CACHE.exists():
        raise FileNotFoundError(f"先にDiscovery exp032 OOFを作る: {BASE_CACHE}")
    data = np.load(BASE_CACHE, allow_pickle=True)
    y = data["y"].astype(int)
    expert_names = tuple(data["names"].tolist())
    experts = {name: np.asarray(data[f"oof_{name}"], dtype=float)
               for name in expert_names}

    # exp029=concat embedding 6本、exp031=そこへE7、exp032=さらにE0b。
    exp029, _, _ = _cross_fitted_meta(y, experts, expert_names[:6], seed)
    exp031, _, _ = _cross_fitted_meta(y, experts, expert_names[:7], seed)
    library = {
        BASE_NAME: np.asarray(data["blend"], dtype=float),
        "exp031": exp031,
        "exp029": exp029,
        **experts,
        "R1_target_text": _load_vector(R1_CACHE, "candidate_oof", y),
        "R4_modern_nca": _load_vector(R4_CACHE, "modern_nca_oof", y),
        "H42_auto_cross": _load_vector(AUTOCROSS_CACHE, "candidate_oof", y),
        "exp035_fixed_knn": _load_vector(R4_CACHE, "fixed_knn_oof", y),
    }
    assert len(library) == 15
    return y, library


def _stability(fold_diagnostics, names):
    sets = [set(d["counts"]) for d in fold_diagnostics]
    jaccards = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            jaccards.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)

    vectors = np.array([
        [d["weights"].get(name, 0.0) for name in names]
        for d in fold_diagnostics
    ])
    cosines = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            denom = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            cosines.append(float(vectors[i] @ vectors[j] / denom) if denom else 0.0)
    frequencies = Counter(name for selected in sets for name in selected)
    return {
        "mean_pairwise_jaccard": float(np.mean(jaccards)),
        "mean_weight_cosine": float(np.mean(cosines)),
        "selected_fold_frequency": dict(sorted(
            frequencies.items(), key=lambda row: (-row[1], row[0])
        )),
        "mean_weights": {
            name: float(vectors[:, idx].mean()) for idx, name in enumerate(names)
        },
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


def run(config: GESConfig, smoke: bool):
    y, library = load_library(LOCKBOX_SEED)
    names = tuple(library)
    effective = replace(config, max_iterations=10) if smoke else config
    folds = StratifiedKFold(
        INNER_FOLDS, shuffle=True, random_state=LOCKBOX_SEED,
    )
    ges_oof = np.full(len(y), np.nan, dtype=float)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(np.zeros(len(y)), y), 1):
        if smoke and fold > 1:
            break
        train_predictions = {name: pred[tr] for name, pred in library.items()}
        valid_predictions = {name: pred[va] for name, pred in library.items()}
        weights, diag = select_ensemble(
            y[tr], train_predictions, BASE_NAME, THS, effective,
        )
        pred = apply_weights(valid_predictions, weights)
        ges_oof[va] = pred
        diag.update({
            "fold": fold,
            "valid_ap": float(average_precision_score(y[va], pred)),
            "base_valid_ap": float(average_precision_score(
                y[va], library[BASE_NAME][va],
            )),
        })
        diagnostics.append(diag)
        used = " / ".join(f"{name}:{weight:.2f}" for name, weight in weights.items()
                          if weight > 0)
        print(f"fold {fold}/{INNER_FOLDS}: iter={diag['selected_iterations']} "
              f"train AP={diag['base_ap']:.4f}->{diag['selected_ap']:.4f} "
              f"valid ΔAP={diag['valid_ap'] - diag['base_valid_ap']:+.4f}")
        print(f"  {used}")

    if smoke:
        done = np.isfinite(ges_oof)
        assert done.sum() > 0
        assert ((ges_oof[done] >= 0) & (ges_oof[done] <= 1)).all()
        print(f"\nSMOKE OK: {done.sum()}件を選択非参照のouter foldで予測。")
        return

    assert np.isfinite(ges_oof).all()
    base = library[BASE_NAME]
    score_rows = {name: _scores(y, pred) for name, pred in library.items()}
    score_rows["nested_GES"] = _scores(y, ges_oof)
    table = pd.DataFrame(score_rows).T
    print("\n=== Discovery OOF library / Nested GES ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    delta = (table.loc["nested_GES", ["auc", "ap", "f1"]]
             - table.loc[BASE_NAME, ["auc", "ap", "f1"]])
    stability = _stability(diagnostics, names)
    errors = _error_delta(
        y, base, ges_oof,
        table.loc[BASE_NAME, "th"], table.loc["nested_GES", "th"],
    )
    print("\n=== exp032への増分 ===")
    print("  " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print(f"  member Jaccard={stability['mean_pairwise_jaccard']:.3f} / "
          f"weight cosine={stability['mean_weight_cosine']:.3f}")
    print("  mean weights:")
    for name, weight in sorted(stability["mean_weights"].items(),
                               key=lambda row: -row[1]):
        if weight > 0:
            print(f"    {name:20s} {weight:.3f}")
    print("  Error delta: " + json.dumps(errors, ensure_ascii=False))

    result = {
        "config": asdict(config),
        "seed": LOCKBOX_SEED,
        "n_discovery": len(y),
        "library_names": list(names),
        "excluded": {
            "R2_TabPFN": "official weight/license token unavailable",
            "H33_multiseed": "742-row OOF uses lockbox labels in fold training",
        },
        "scores": {key: {metric: float(value) for metric, value in row.items()}
                   for key, row in score_rows.items()},
        "delta_vs_exp032": {key: float(value) for key, value in delta.items()},
        "stability": stability,
        "errors": errors,
        "fold_diagnostics": diagnostics,
        "lockbox_opened": False,
    }
    np.savez_compressed(
        OUT_NPZ, y=y, exp032=base, nested_ges=ges_oof,
        names=np.array(names),
        **{f"library_{name}": pred for name, pred in library.items()},
    )
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"保存: {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(GESConfig(), smoke=args.smoke)
