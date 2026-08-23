"""exp032(8エキスパート)に R4 structured-only ModernNCA を9本目として足し、提出を作る。

R4はdeep-researchロードマップ検証(2026-08-15/16)では単発discovery seedでΔAP -0.0115
としてREJECTだったが、2026-08-23にdiscovery固定分割・内側5foldの分割seedだけを5本
振ったところ全て正(平均ΔAP +0.0201, 5/5)だった。単体discoveryは通過したが、
holdout_check.py相当の完全検証やlockboxでの確認はまだ済んでいない。
本命(submission_exp032_seed42_20260815.csv)を置き換えるものではなく、比較用の
提出候補として作る。

ベース構成は exp032:
  E4=[組織図;企業概要] embedding、E7=cross LR、E0b=linear(C=0.03)
9本目 E10_modern_nca は R4検証と同じ構造化47列だけを入力にする
(dataset.build_matrices の基本列。TF-IDF/embedding/LLMスコアは使わない)。

実行:
  OMP_NUM_THREADS=4 python3 exp/make_submission_modern_nca.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CAT_COLS, build_matrices  # noqa: E402
from ensemble_experts import (  # noqa: E402
    _resolve_threshold, _scores, blend_oof, compute_expert_preds,
)
from modern_nca import ModernNCAConfig, fit_predict_fold  # noqa: E402

E10_NAME = "E10_modern_nca"


def modern_nca_oof_and_test(X, y, Xte, seed, config):
    """既存expertと同じ5fold構造でOOF(742)とtest(800)予測を作る。

    1foldにつき1回だけ学習し、valid+testをまとめて予測することで
    OOF側とtest側で同一モデルになるようにする(2回学習してseedが同じでも
    別インスタンスになる揺れを避ける)。
    """
    oof = np.zeros(len(y))
    te_parts = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"  E10 ModernNCA fold {fold}/5 ...", flush=True)
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        combined = pd.concat([Xva, Xte], axis=0, ignore_index=True)
        pred, _diag = fit_predict_fold(
            Xtr, y[tr], combined, None, tuple(CAT_COLS), config,
            seed=seed + fold * 10000,
        )
        oof[va] = pred[: len(va)]
        te_parts.append(pred[len(va):])
    te = np.mean(te_parts, axis=0)
    return oof, te


def run(seed, alpha, submit, no_cache, test_pos, threshold, output):
    y, oof8, te8 = compute_expert_preds(
        seed, cache=not no_cache, concat_embed=True, cross=True,
        e7_model="lr", linear=True, linear_c=0.03,
    )

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X_all, y_check, Xte_all = build_matrices(train, test)
    assert np.array_equal(y, y_check)

    config = ModernNCAConfig(epochs=80, threads=4)
    print("E10(structured-only ModernNCA / R4)を9本目として学習")
    nca_oof, nca_te = modern_nca_oof_and_test(X_all, y, Xte_all, seed, config)

    oof9 = {**oof8, E10_NAME: nca_oof}
    te9 = {**te8, E10_NAME: nca_te}
    experts = tuple(oof9.keys())

    rows = {n: _scores(y, oof9[n]) for n in experts}
    blend, blend_te, w, alphas = blend_oof(y, oof9, te9, seed, alpha,
                                           experts=experts, target="zero")
    key = "-- meta (alpha=auto)" if alpha is None else f"-- meta (alpha={alpha})"
    rows[key] = _scores(y, blend)

    table = pd.DataFrame(rows).T
    print(f"\n=== OOFスコア (seed={seed}) ===")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if alpha is None:
        print(f"外側foldごとに選ばれたalpha: {alphas}")

    print("\n=== 学習された重み (5fold平均, 非負) ===")
    for n, v in sorted(zip(experts, w), key=lambda x: -x[1]):
        bar = "#" * int(round(v * 40))
        print(f"  {n:16s} {v:6.3f} {bar}")

    base_key = "-- meta (alpha=auto)" if alpha is None else key
    print(f"\n8experts(exp032) -> 9experts(+R4):")

    if submit:
        m = table.loc[key]
        th, _ = _resolve_threshold(blend, blend_te, y, m.th, test_pos, threshold)
        label = (blend_te >= th).astype(int)
        sample = pd.read_csv("data/sample_submit.csv", header=None,
                             names=["企業ID", "購入フラグ"])
        pred = pd.DataFrame({"企業ID": test["企業ID"].values, "pred": label})
        sub = sample[["企業ID"]].merge(pred, on="企業ID", how="left")
        assert sub["pred"].notna().all(), "予測欠損あり"
        tag = "auto" if alpha is None else str(alpha)
        out = output or (
            f"submission/submission_exp032_plus_modernca_seed{seed}_a{tag}.csv"
        )
        if os.path.exists(out):
            raise FileExistsError(f"既存の提出物は上書きしない: {out}")
        sub.assign(pred=sub["pred"].astype(int)).to_csv(
            out, index=False, header=False, lineterminator="\n")
        print(f"保存: {out} 正例={int(label.sum())} (th={th:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=None,
                   help="固定するなら数値、省略で自動選択")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--output")
    p.add_argument("--no-cache", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--test-pos", type=int)
    g.add_argument("--threshold", type=float, help="閾値を直接指定")
    args = p.parse_args()
    run(args.seed, args.alpha, args.submit, args.no_cache,
        args.test_pos, args.threshold, args.output)
