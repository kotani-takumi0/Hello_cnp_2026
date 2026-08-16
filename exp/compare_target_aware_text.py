"""R1: target-aware DX展望 expertをdiscovery内でexp032と比較する。

通常実行では固定discovery 482件だけを使い、lockbox 260件は読み出しも評価もしない。
各外側foldのtrain labelだけでencoderをcontrastive fine-tuneし、そのfoldのvalidationを
予測する。候補OOFをexp032へ9本目として追加し、単体性能だけでなく増分を測る。

まず配管確認:

  OMP_NUM_THREADS=4 python3 exp/compare_target_aware_text.py --smoke

Discovery 1-seed正式スクリーニング:

  OMP_NUM_THREADS=4 python3 exp/compare_target_aware_text.py

``--smoke`` は最初のouter foldを2 optimizer stepだけ学習する。性能判定には使わない。
通常実行も最初は8 step/foldに固定し、CPUで仮説を安価に殺す。結果を見てstep数を
調整することはせず、この事前固定値で昇格しなければ最小構成は終了する。
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_experts import THS, _scores  # noqa: E402
from lockbox_error_analysis import (  # noqa: E402
    INNER_FOLDS, LOCKBOX_SEED, _cross_fitted_meta, discovery_oof, fixed_split,
)
from target_aware_text import (  # noqa: E402
    DEFAULT_MODEL, TargetAwareConfig, fit_predict_fold,
)
from text_features import TEXT_COL  # noqa: E402


NAME = "E10_target_text"
OUT_NPZ = Path("exp/_r1_target_text_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_r1_target_text_discovery_seed20260815.json")


def crossfit_candidate(texts: np.ndarray, y: np.ndarray,
                       config: TargetAwareConfig, seed: int,
                       smoke: bool = False):
    folds = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed,
    )
    oof = np.full(len(y), np.nan, dtype=float)
    diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(texts, y), 1):
        if smoke and fold > 1:
            break
        print(f"target-aware fold {fold}/{INNER_FOLDS}: "
              f"train={len(tr)} valid={len(va)}", flush=True)
        t0 = time.time()
        pred, _, diag = fit_predict_fold(
            texts[tr], y[tr], texts[va], config,
            seed=seed + fold * 10000,
        )
        oof[va] = pred
        diag.update({"fold": fold, "seconds": time.time() - t0})
        diagnostics.append(diag)
        if len(np.unique(y[va])) == 2:
            print(f"  AP={average_precision_score(y[va], pred):.4f} "
                  f"AUC={roc_auc_score(y[va], pred):.4f} "
                  f"loss={diag['loss']:.4f} steps={diag['steps']} "
                  f"time={diag['seconds']:.0f}s", flush=True)
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


def run(config: TargetAwareConfig, smoke: bool, refresh_base: bool):
    train = pd.read_csv("data/train.csv")
    y_all = train["購入フラグ"].to_numpy(dtype=int)
    discovery, _ = fixed_split(y_all)
    texts = train.iloc[discovery][TEXT_COL].fillna("").astype(str).to_numpy()
    y = y_all[discovery]

    effective = replace(config, max_steps=2) if smoke else config
    print(f"discovery={len(discovery)} positive_rate={y.mean():.4f}")
    print("config=" + json.dumps(asdict(effective), ensure_ascii=False))
    candidate_oof, diagnostics = crossfit_candidate(
        texts, y, effective, LOCKBOX_SEED, smoke=smoke,
    )
    if smoke:
        done = ~np.isnan(candidate_oof)
        assert done.sum() > 0 and np.isfinite(candidate_oof[done]).all()
        assert ((candidate_oof[done] >= 0) & (candidate_oof[done] <= 1)).all()
        print(f"\nSMOKE OK: {done.sum()}件をfold外予測。性能判定はしない。")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
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
        "E3_dx_text": _scores(y, base_oof["E3_dx_text"]),
        NAME: _scores(y, candidate_oof),
        "exp032": _scores(y, base_blend),
        "exp032+target_text": _scores(y, candidate_blend),
    }
    table = pd.DataFrame(score_rows).T
    print("\n=== Discovery OOF ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    delta = (table.loc["exp032+target_text", ["auc", "ap", "f1"]]
             - table.loc["exp032", ["auc", "ap", "f1"]])
    corrs = {
        "E3_dx_text": float(spearmanr(candidate_oof,
                                      base_oof["E3_dx_text"]).statistic),
        "E4_org_text": float(spearmanr(candidate_oof,
                                       base_oof["E4_org_text"]).statistic),
        "exp032": float(spearmanr(candidate_oof, base_blend).statistic),
    }
    errors = _error_delta(
        y, base_blend, candidate_blend,
        table.loc["exp032", "th"], table.loc["exp032+target_text", "th"],
    )
    target_weight = float(weights[-1])

    print("\n=== exp032への増分 ===")
    print("  " + " / ".join(f"Δ{k.upper()} {v:+.4f}" for k, v in delta.items()))
    print(f"  target-aware meta weight={target_weight:.4f}")
    print("  Spearman: " + " / ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    print("  Error delta: " + json.dumps(errors, ensure_ascii=False))
    print(f"  alpha per fold: {alphas}")

    result = {
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--last-layers", type=int, default=2)
    parser.add_argument("--pairs-per-anchor", type=int, default=1)
    parser.add_argument("--head-c", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8,
                        help="foldごとのoptimizer step上限（最小実験の事前固定値8）")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    cfg = TargetAwareConfig(
        model_name=args.model,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        learning_rate=args.lr,
        train_last_n_layers=args.last_layers,
        pairs_per_anchor=args.pairs_per_anchor,
        head_c=args.head_c,
        threads=args.threads,
        local_files_only=not args.allow_download,
        max_steps=args.max_steps,
    )
    run(cfg, smoke=args.smoke, refresh_base=args.refresh_base)
