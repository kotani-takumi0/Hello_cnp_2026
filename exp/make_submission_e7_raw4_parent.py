"""exp032 の E7 を「アンケート4問個別 + 財務親5軸」に差し替えて提出を作る。

現行 E7 は Q7/Q10 と Q2/Q4 を等重みの不満軸へ集約している。この候補では
Q2/Q4/Q7/Q10 を個別列として LR に渡し、fold 内で設問別の重みを学習させる。
事後診断で純増がほぼ無かった設問×財務の交互作用5本は使用しない。

ベース構成は exp032:
  E4=[組織図;企業概要] embedding、E7=cross LR、E0b=linear(C=0.03)

実行:
  python3 exp/make_submission_e7_raw4_parent.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cross_features import RATIO_COLS  # noqa: E402
from ensemble_experts import (  # noqa: E402
    E0B_NAME, E7_NAME, EXPERTS, OUT_DIR, _cross_lr, _scores, blend_oof,
    build_features, compute_expert_preds,
)

UNINSTALLED_Q7 = 3.3
SURVEY_ITEMS = ("アンケート２", "アンケート４", "アンケート７", "アンケート１０")
SURVEY_FEATURES = tuple(f"不満_{c}" for c in SURVEY_ITEMS)
RAW4_PARENT_COLS = SURVEY_FEATURES + tuple(RATIO_COLS)
BASE_MEMBERS = EXPERTS + (E7_NAME, E0B_NAME)
DEFAULT_OUTPUT = "submission/submission_exp032_e7_raw4_parent_seed42.csv"


def _raw4_parent_frame(raw, features):
    """4設問を不満方向へ反転し、既存の財務親5軸と並べる。"""
    out = pd.DataFrame(index=raw.index)
    for col in SURVEY_ITEMS:
        values = raw[col].astype(float)
        if col == "アンケート７":
            values = values.fillna(UNINSTALLED_Q7)
        out[f"不満_{col}"] = 6.0 - values
    for col in RATIO_COLS:
        out[col] = features[col].astype(float)
    return out[list(RAW4_PARENT_COLS)]


def _candidate_predictions(seed, cache=True):
    path = os.path.join(OUT_DIR, f"_e7_raw4_parent_seed{seed}.npz")
    if cache and os.path.exists(path):
        data = np.load(path)
        return data["y"], data["oof"], data["test"]

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xte = build_features(train, test, cross=True)
    candidate = _raw4_parent_frame(train, X)
    candidate_test = _raw4_parent_frame(test, Xte)

    oof = np.zeros(len(y))
    test_parts = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(candidate, y), 1):
        print(f"  E7 raw4 parent fold {fold}/5 ...", flush=True)
        p, q = _cross_lr(
            candidate.iloc[tr], y[tr], candidate.iloc[va], candidate_test
        )
        oof[va] = p
        test_parts.append(q)

    test_pred = np.mean(test_parts, axis=0)
    np.savez(path, y=y, oof=oof, test=test_pred)
    return y, oof, test_pred


def _base_predictions(seed):
    path = os.path.join(
        OUT_DIR, f"_experts_seed{seed}_concatemb_crosslr_lin0.03.npz"
    )
    if not os.path.exists(path):
        compute_expert_preds(
            seed, concat_embed=True, cross=True, e7_model="lr",
            linear=True, linear_c=0.03,
        )
    data = np.load(path)
    y = data["y"]
    oof = {name: data[f"oof_{name}"] for name in BASE_MEMBERS}
    test = {name: data[f"te_{name}"] for name in BASE_MEMBERS}
    return y, oof, test


def run(seed, output, alpha=None, no_cache=False):
    y, base_oof, base_test = _base_predictions(seed)
    yy, candidate_oof, candidate_test = _candidate_predictions(
        seed, cache=not no_cache
    )
    assert np.array_equal(y, yy)

    oof = {**base_oof, E7_NAME: candidate_oof}
    test = {**base_test, E7_NAME: candidate_test}
    base_blend, _, _, _ = blend_oof(
        y, base_oof, base_test, seed, alpha=alpha, experts=BASE_MEMBERS
    )
    blend, blend_test, weights, alphas = blend_oof(
        y, oof, test, seed, alpha=alpha, experts=BASE_MEMBERS
    )

    base_score = _scores(y, base_blend)
    score = _scores(y, blend)
    threshold = score["th"]
    labels = (blend_test >= threshold).astype(int)

    raw_test = pd.read_csv("data/test.csv")
    sample = pd.read_csv(
        "data/sample_submit.csv", header=None, names=["企業ID", "購入フラグ"]
    )
    pred = pd.DataFrame({"企業ID": raw_test["企業ID"].values, "pred": labels})
    submission = sample[["企業ID"]].merge(pred, on="企業ID", how="left")
    assert len(submission) == len(raw_test) == 800
    assert submission["pred"].notna().all()
    assert set(submission["pred"].unique()) <= {0, 1}
    submission.assign(pred=submission["pred"].astype(int)).to_csv(
        output, index=False, header=False, lineterminator="\n"
    )

    print("\n=== OOF comparison ===")
    print(pd.DataFrame({"exp032": base_score, "E7_raw4_parent": score}).T.to_string(
        float_format=lambda value: f"{value:.4f}"
    ))
    print(f"\nmeta alpha: {'auto' if alpha is None else alpha}")
    print("meta alpha by fold:", alphas)
    print("meta weights (5fold mean):")
    for name, value in sorted(zip(BASE_MEMBERS, weights), key=lambda x: -x[1]):
        print(f"  {name:16s} {value:.4f}")
    print(
        f"\n保存: {output}\n"
        f"行数={len(submission)} / 正例={int(labels.sum())} "
        f"({labels.mean():.2%}) / OOF閾値={threshold:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output, args.alpha, args.no_cache)
