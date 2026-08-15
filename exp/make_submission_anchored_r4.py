"""H43 challenger: exp032 93% + raw ModernNCA 7% の提出を作る。

Discovery nested選択lambda (0.10/0/0.10/0.10/0.05) の平均0.07を固定する。
これはH43の事前ACCEPT条件には届かなかった探索提出であり、Public 0.80000の
exp032本命を置き換えない。

  MPLCONFIGDIR=/tmp/matplotlib-h43 OMP_NUM_THREADS=4 \
    python3 exp/make_submission_anchored_r4.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CAT_COLS, build_matrices  # noqa: E402
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, _scores, blend_oof, compute_expert_preds,
)
from lockbox_error_analysis import INNER_FOLDS, LOCKBOX_SEED  # noqa: E402
from modern_nca import ModernNCAConfig, fit_predict_fold  # noqa: E402


LAMBDA = 0.07
BASE_SEED = 42
MODERN_NCA_SEED = LOCKBOX_SEED
BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
MODERN_CACHE = Path("exp/_h43_modern_nca_full_seed20260815.npz")
DEFAULT_OUTPUT = Path(
    "submission/submission_exp032_anchor_r4_lam007_seed20260815.csv"
)


def _modern_oof_test(no_cache: bool):
    if MODERN_CACHE.exists() and not no_cache:
        data = np.load(MODERN_CACHE)
        print(f"ModernNCA cache: {MODERN_CACHE}")
        return data["y"].astype(int), data["oof"], data["test"]

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xtest = build_matrices(train, test)
    config = ModernNCAConfig()
    folds = StratifiedKFold(
        INNER_FOLDS, shuffle=True, random_state=MODERN_NCA_SEED,
    )
    oof = np.zeros(len(y), dtype=float)
    test_parts = []
    # 同じmodelでvalidとtestを一度に推論し、foldごとのtest予測を平均する。
    evaluation = pd.concat([X, Xtest], axis=0, ignore_index=True)
    for fold, (tr, va) in enumerate(folds.split(X, y), 1):
        print(f"ModernNCA full fold {fold}/{INNER_FOLDS} ...", flush=True)
        fold_evaluation = pd.concat(
            [X.iloc[va], evaluation.iloc[len(X):]],
            axis=0, ignore_index=True,
        )
        pred, diag = fit_predict_fold(
            X.iloc[tr], y[tr], fold_evaluation, None, tuple(CAT_COLS), config,
            seed=MODERN_NCA_SEED + fold * 10000,
        )
        oof[va] = pred[:len(va)]
        test_parts.append(pred[len(va):])
        print(f"  loss={diag['initial_loss']:.4f}->{diag['final_loss']:.4f}",
              flush=True)
    test_pred = np.mean(test_parts, axis=0)
    np.savez_compressed(MODERN_CACHE, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def _base_oof_test():
    y, oof, test = compute_expert_preds(
        BASE_SEED, concat_embed=True, cross=True, e7_model="lr",
        linear=True, linear_c=0.03,
    )
    base_oof, base_test, _, _ = blend_oof(
        y, oof, test, BASE_SEED, alpha=None, experts=BASE_MEMBERS,
    )
    return y, base_oof, base_test


def run(output: Path, no_cache: bool):
    if output.exists():
        raise FileExistsError(f"既存の提出物は上書きしない: {output}")
    y, modern_oof, modern_test = _modern_oof_test(no_cache)
    yy, base_oof, base_test = _base_oof_test()
    assert np.array_equal(y, yy)
    blend_oof_pred = (1.0 - LAMBDA) * base_oof + LAMBDA * modern_oof
    blend_test = (1.0 - LAMBDA) * base_test + LAMBDA * modern_test
    score = _scores(y, blend_oof_pred)
    base_score = _scores(y, base_oof)
    threshold = float(score["th"])
    labels = (blend_test >= threshold).astype(int)

    raw_test = pd.read_csv("data/test.csv")
    sample = pd.read_csv(
        "data/sample_submit.csv", header=None, names=["企業ID", "購入フラグ"],
    )
    pred = pd.DataFrame({"企業ID": raw_test["企業ID"].values, "pred": labels})
    submission = sample[["企業ID"]].merge(pred, on="企業ID", how="left")
    assert submission.shape == (800, 2)
    assert submission["企業ID"].equals(sample["企業ID"])
    assert submission["pred"].notna().all()
    assert set(submission["pred"].unique()) <= {0, 1}
    submission.assign(pred=submission["pred"].astype(int)).to_csv(
        output, index=False, header=False, lineterminator="\n",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    print("\n=== Full OOF reference ===")
    print(pd.DataFrame({"exp032": base_score, "anchored_R4": score}).T.to_string(
        float_format=lambda value: f"{value:.4f}",
    ))
    print(f"ΔAUC={score['auc'] - base_score['auc']:+.4f} "
          f"ΔAP={score['ap'] - base_score['ap']:+.4f} "
          f"ΔF1={score['f1'] - base_score['f1']:+.4f}")
    print(f"保存: {output}")
    print(f"lambda={LAMBDA:.2f} threshold={threshold:.3f} "
          f"positive={int(labels.sum())}/800 SHA256={digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    run(args.output, args.no_cache)
