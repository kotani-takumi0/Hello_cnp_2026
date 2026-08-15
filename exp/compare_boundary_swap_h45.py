"""H45: exp032のthreshold境界だけをR4/R1で固定K swapするnested評価。

全体順位を混ぜて陽性数を変えるH43/H44から一段保守的にし、exp032のouter-train
thresholdで決まる境界の内側B件だけを候補にする。候補支持度はexp032/R4/R1の
percentile rank blend、swap数mはinner foldで選ぶ。必ずm件を各方向に交換する。

  python3 exp/compare_boundary_swap_h45.py --smoke
  python3 exp/compare_boundary_swap_h45.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_anchored_r4 import _best_threshold  # noqa: E402
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402


R4_CACHE = Path("exp/_r4_modern_nca_discovery_seed20260815.npz")
R1_CACHE = Path("exp/_r1_target_text_discovery_seed20260815.npz")
OUT_NPZ = Path("exp/_h45_boundary_swap_discovery_seed20260815.npz")
OUT_JSON = Path("exp/_h45_boundary_swap_discovery_seed20260815.json")

# Baseを強く残す。合計0.30まで補助モデルを使う低自由度grid。
WEIGHTS = (
    (0.00, 0.00), (0.10, 0.00), (0.20, 0.00),
    (0.00, 0.10), (0.00, 0.20),
    (0.05, 0.05), (0.10, 0.10), (0.15, 0.15),
)
BOUNDARY_WIDTHS = (20, 40, 60)
SWAPS = (0, 1, 2, 3, 5, 8, 10)
AP_TOLERANCE = 0.002
INNER_SELECTION_FOLDS = 3
MIN_TRANSFER_F1_GAIN = 0.005
MAX_SINGLE_FOLD_F1_DROP = 0.02


def _rank(x):
    x = np.asarray(x, dtype=float)
    return rankdata(x, method="average") / len(x)


def _support(base, r4, r1, w4, w1):
    return (1.0 - w4 - w1) * _rank(base) + w4 * _rank(r4) + w1 * _rank(r1)


def _swap_labels(base, support, threshold, width, m):
    base_label = (base >= threshold).astype(int)
    if m == 0:
        return base_label
    neg = np.flatnonzero(base_label == 0)
    pos = np.flatnonzero(base_label == 1)
    neg = neg[np.argsort(np.abs(base[neg] - threshold), kind="mergesort")[:width]]
    pos = pos[np.argsort(np.abs(base[pos] - threshold), kind="mergesort")[:width]]
    m_eff = min(m, len(neg), len(pos))
    out = base_label.copy()
    add = neg[np.argsort(-support[neg], kind="mergesort")[:m_eff]]
    drop = pos[np.argsort(support[pos], kind="mergesort")[:m_eff]]
    out[add] = 1
    out[drop] = 0
    assert int((out == 1).sum()) == int((base_label == 1).sum())
    return out


def _select(y, base, r4, r1, seed):
    inner = StratifiedKFold(INNER_SELECTION_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for w4, w1 in WEIGHTS:
        for width in BOUNDARY_WIDTHS:
            for m in SWAPS:
                aps, f1s, base_f1s = [], [], []
                for tr, va in inner.split(np.zeros(len(y)), y):
                    th, _ = _best_threshold(y[tr], base[tr])
                    b_lab = (base[va] >= th).astype(int)
                    s = _support(base[va], r4[va], r1[va], w4, w1)
                    c_lab = _swap_labels(base[va], s, th, width, m)
                    aps.append(float(average_precision_score(y[va], s)))
                    f1s.append(float(f1_score(y[va], c_lab)))
                    base_f1s.append(float(f1_score(y[va], b_lab)))
                rows.append({
                    "w4": w4, "w1": w1, "width": width, "swaps": m,
                    "mean_ap": float(np.mean(aps)), "mean_f1": float(np.mean(f1s)),
                    "mean_base_f1": float(np.mean(base_f1s)),
                    "ap_per_fold": aps, "f1_per_fold": f1s,
                })
    base_ap = max(r["mean_ap"] for r in rows if r["w4"] == 0 and r["w1"] == 0)
    for r in rows:
        r["ap_delta_vs_base"] = r["mean_ap"] - base_ap
        r["f1_delta_vs_base"] = r["mean_f1"] - r["mean_base_f1"]
        r["eligible"] = r["mean_ap"] >= base_ap - AP_TOLERANCE
    eligible = [r for r in rows if r["eligible"]]
    choice = max(eligible, key=lambda r: (
        r["mean_f1"], r["mean_ap"], -r["swaps"], -r["w4"] - r["w1"],
    ))
    return choice, rows


def run(smoke=False):
    for path in (R4_CACHE, R1_CACHE):
        if not path.exists():
            raise FileNotFoundError(path)
    a = np.load(R4_CACHE); b = np.load(R1_CACHE)
    y = a["y"].astype(int); base = a["base_blend"].astype(float)
    r4 = a["modern_nca_oof"].astype(float); r1 = b["candidate_oof"].astype(float)
    assert np.array_equal(y, b["y"])
    assert np.max(np.abs(base - b["base_blend"])) == 0

    folds = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=LOCKBOX_SEED)
    cand_score = np.full(len(y), np.nan); base_label = np.full(len(y), -1, int)
    cand_label = np.full(len(y), -1, int); diagnostics = []
    for fold, (tr, va) in enumerate(folds.split(np.zeros(len(y)), y), 1):
        if smoke and fold > 1: break
        choice, grid = _select(y[tr], base[tr], r4[tr], r1[tr], LOCKBOX_SEED + fold * 10000)
        th, _ = _best_threshold(y[tr], base[tr])
        s = _support(base[va], r4[va], r1[va], choice["w4"], choice["w1"])
        bl = (base[va] >= th).astype(int)
        cl = _swap_labels(base[va], s, th, choice["width"], choice["swaps"])
        base_label[va] = bl; cand_label[va] = cl; cand_score[va] = s
        diagnostics.append({
            "fold": fold, "threshold": th, **{k: choice[k] for k in ("w4", "w1", "width", "swaps")},
            "positive_budget": int(bl.sum()), "swap_actual": int(((bl == 0) & (cl == 1)).sum()),
            "outer_base_ap": float(average_precision_score(y[va], base[va])),
            "outer_candidate_ap": float(average_precision_score(y[va], s)),
            "outer_base_f1": float(f1_score(y[va], bl)),
            "outer_candidate_f1": float(f1_score(y[va], cl)),
            "inner_choice": choice, "inner_grid": grid,
        })
        print(f"fold {fold}/{INNER_FOLDS}: w4={choice['w4']:.2f} w1={choice['w1']:.2f} "
              f"B={choice['width']} m={choice['swaps']} K={int(bl.sum())} "
              f"ΔAP={diagnostics[-1]['outer_candidate_ap'] - diagnostics[-1]['outer_base_ap']:+.4f} "
              f"ΔF1={diagnostics[-1]['outer_candidate_f1'] - diagnostics[-1]['outer_base_f1']:+.4f}")
    if smoke:
        done = np.isfinite(cand_score); assert done.sum() > 0
        assert np.array_equal(cand_label[done].sum(), base_label[done].sum())
        print(f"SMOKE OK: {done.sum()}件、固定K")
        return

    base_auc = roc_auc_score(y, base); cand_auc = roc_auc_score(y, cand_score)
    base_ap = average_precision_score(y, base); cand_ap = average_precision_score(y, cand_score)
    base_f1 = f1_score(y, base_label); cand_f1 = f1_score(y, cand_label)
    deltas = [d["outer_candidate_f1"] - d["outer_base_f1"] for d in diagnostics]
    passed = (cand_ap - base_ap >= -AP_TOLERANCE and cand_f1 - base_f1 >= MIN_TRANSFER_F1_GAIN
              and min(deltas) >= -MAX_SINGLE_FOLD_F1_DROP)
    verdict = "PROMISING" if passed else "REJECT"
    errors = {
        "base_errors": int((base_label != y).sum()), "candidate_errors": int((cand_label != y).sum()),
        "fn_rescued": int(((y == 1) & (base_label == 0) & (cand_label == 1)).sum()),
        "new_fn": int(((y == 1) & (base_label == 1) & (cand_label == 0)).sum()),
        "fp_removed": int(((y == 0) & (base_label == 1) & (cand_label == 0)).sum()),
        "new_fp": int(((y == 0) & (base_label == 0) & (cand_label == 1)).sum()),
        "zero_to_one": int(((base_label == 0) & (cand_label == 1)).sum()),
        "one_to_zero": int(((base_label == 1) & (cand_label == 0)).sum()),
    }
    print("\n=== H45 boundary swap ===")
    print(pd.DataFrame({"exp032": [base_auc, base_ap, base_f1], "H45": [cand_auc, cand_ap, cand_f1]}, index=["AUC","AP","budgeted F1"]).to_string(float_format=lambda x:f"{x:.4f}"))
    print(f"ΔAUC={cand_auc-base_auc:+.4f} ΔAP={cand_ap-base_ap:+.4f} ΔF1={cand_f1-base_f1:+.4f}")
    print("fold ΔF1:", [round(x, 4) for x in deltas]); print("errors:", json.dumps(errors, ensure_ascii=False)); print("verdict=", verdict)
    result = {"method": "boundary-only fixed-K swap with rank(exp032,R4,R1)", "weights": [list(x) for x in WEIGHTS], "boundary_widths": list(BOUNDARY_WIDTHS), "swaps": list(SWAPS), "ap_tolerance": AP_TOLERANCE, "acceptance": {"min_f1_gain": MIN_TRANSFER_F1_GAIN, "max_fold_drop": MAX_SINGLE_FOLD_F1_DROP}, "scores": {"exp032": {"auc": base_auc, "ap": base_ap, "f1": base_f1}, "H45": {"auc": cand_auc, "ap": cand_ap, "f1": cand_f1}}, "deltas": {"auc": cand_auc-base_auc, "ap": cand_ap-base_ap, "f1": cand_f1-base_f1}, "fold_f1_deltas": deltas, "errors": errors, "diagnostics": diagnostics, "verdict": verdict, "lockbox_opened": False}
    np.savez_compressed(OUT_NPZ, y=y, exp032=base, candidate=cand_score, exp032_label=base_label, candidate_label=cand_label)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {OUT_NPZ} / {OUT_JSON}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); run(p.parse_args().smoke)
