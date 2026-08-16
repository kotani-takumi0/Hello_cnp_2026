"""H44: exp032と同じ陽性数に固定したModernNCA rank blend提出を作る。

Discovery nested 5-foldで選ばれたlambda (0.15/0.15/0.10/0.10/0.10) の
fold平均0.12を固定する。testのKは本命exp032提出の陽性数をそのまま使う。

  MPLCONFIGDIR=/tmp/matplotlib-h44 python3 \
    exp/make_submission_prevalence_rank_blend.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_prevalence_rank_blend import (  # noqa: E402
    _blend_rank, _top_k_label,
)
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, _scores, blend_oof, compute_expert_preds,
)


LAMBDA = 0.12
BASE_SEED = 42
BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
MODERN_CACHE = Path("exp/_h43_modern_nca_full_seed20260815.npz")
BASE_SUBMISSION = Path("submission/submission_exp032_seed42_20260815.csv")
DEFAULT_OUTPUT = Path(
    "submission/submission_exp032_rankblend_r4_lam012_top236_seed20260815.csv"
)


def _base_oof_test():
    y, oof, test = compute_expert_preds(
        BASE_SEED, concat_embed=True, cross=True, e7_model="lr",
        linear=True, linear_c=0.03,
    )
    base_oof, base_test, _, _ = blend_oof(
        y, oof, test, BASE_SEED, alpha=None, experts=BASE_MEMBERS,
    )
    return y, base_oof, base_test


def run(output: Path):
    if output.exists():
        raise FileExistsError(f"既存の提出物は上書きしない: {output}")
    if not MODERN_CACHE.exists():
        raise FileNotFoundError(
            f"先にModernNCA full OOF/test cacheを作る: {MODERN_CACHE}"
        )
    if not BASE_SUBMISSION.exists():
        raise FileNotFoundError(f"本命提出が見つからない: {BASE_SUBMISSION}")

    modern = np.load(MODERN_CACHE)
    y, base_oof, base_test = _base_oof_test()
    assert np.array_equal(y, modern["y"].astype(int))
    modern_oof = np.asarray(modern["oof"], dtype=float)
    modern_test = np.asarray(modern["test"], dtype=float)
    assert base_oof.shape == modern_oof.shape == y.shape
    assert base_test.shape == modern_test.shape == (800,)

    candidate_oof = _blend_rank(base_oof, modern_oof, LAMBDA)
    candidate_test = _blend_rank(base_test, modern_test, LAMBDA)
    base_score = _scores(y, base_oof)
    candidate_score = _scores(y, candidate_oof)

    # OOF参考もbaseのbest thresholdが作る陽性数を候補へ移植する。
    base_oof_label = (base_oof >= float(base_score["th"])).astype(int)
    candidate_oof_label = _top_k_label(candidate_oof, int(base_oof_label.sum()))
    base_oof_f1 = float(
        2 * np.sum((y == 1) & (base_oof_label == 1)) /
        (2 * np.sum((y == 1) & (base_oof_label == 1))
         + np.sum((y == 0) & (base_oof_label == 1))
         + np.sum((y == 1) & (base_oof_label == 0)))
    )
    candidate_oof_f1 = float(
        2 * np.sum((y == 1) & (candidate_oof_label == 1)) /
        (2 * np.sum((y == 1) & (candidate_oof_label == 1))
         + np.sum((y == 0) & (candidate_oof_label == 1))
         + np.sum((y == 1) & (candidate_oof_label == 0)))
    )

    base_submission = pd.read_csv(
        BASE_SUBMISSION, header=None, names=["企業ID", "購入フラグ"],
    )
    k = int(base_submission["購入フラグ"].sum())
    deployment_base_oof_label = _top_k_label(base_oof, k)
    deployment_candidate_oof_label = _top_k_label(candidate_oof, k)
    deployment_base_oof_f1 = float(
        2 * np.sum((y == 1) & (deployment_base_oof_label == 1)) /
        (2 * np.sum((y == 1) & (deployment_base_oof_label == 1))
         + np.sum((y == 0) & (deployment_base_oof_label == 1))
         + np.sum((y == 1) & (deployment_base_oof_label == 0)))
    )
    deployment_candidate_oof_f1 = float(
        2 * np.sum((y == 1) & (deployment_candidate_oof_label == 1)) /
        (2 * np.sum((y == 1) & (deployment_candidate_oof_label == 1))
         + np.sum((y == 0) & (deployment_candidate_oof_label == 1))
         + np.sum((y == 1) & (deployment_candidate_oof_label == 0)))
    )
    base_top_k = _top_k_label(base_test, k)
    assert np.array_equal(
        base_top_k, base_submission["購入フラグ"].to_numpy(dtype=int)
    ), "base提出とbase score上位Kが一致しない"
    labels = _top_k_label(candidate_test, k)
    assert int(labels.sum()) == k

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

    zero_to_one = (base_top_k == 0) & (labels == 1)
    one_to_zero = (base_top_k == 1) & (labels == 0)
    assert int(zero_to_one.sum()) == int(one_to_zero.sum())
    changed_ids = {
        "zero_to_one": raw_test.loc[zero_to_one, "企業ID"].astype(int).tolist(),
        "one_to_zero": raw_test.loc[one_to_zero, "企業ID"].astype(int).tolist(),
    }

    print("\n=== Full OOF reference (lambdaの再選択には使わない) ===")
    print(pd.DataFrame({"exp032": base_score, "fixed_K_rank_blend": candidate_score})
          .T.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"same-K OOF F1: exp032={base_oof_f1:.4f} "
          f"candidate={candidate_oof_f1:.4f} "
          f"delta={candidate_oof_f1 - base_oof_f1:+.4f}")
    print(f"deployment-K({k}) OOF F1: exp032={deployment_base_oof_f1:.4f} "
          f"candidate={deployment_candidate_oof_f1:.4f} "
          f"delta={deployment_candidate_oof_f1 - deployment_base_oof_f1:+.4f}")
    print(f"保存: {output}")
    print(f"lambda={LAMBDA:.2f} positive={int(labels.sum())}/800 "
          f"swaps_each_direction={int(zero_to_one.sum())} SHA256={digest}")
    print("changed IDs: " + str(changed_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)
