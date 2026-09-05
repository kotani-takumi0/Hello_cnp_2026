"""H97 の最良セル（S=3 ノイズ整合 + cbpe_platt 閾値）を提出物にする。

`documents/h97_noise_matched_stacking_20260901.md` の 11rep strict holdout で、
現本番（S=1 + half_f1）0.8213 / 最悪rep 0.7903 に対し
S=3 + cbpe_platt は **0.8294 / 最悪rep 0.7969** と平均・最悪の両方で勝った。

構成はホールドアウトで検証したものをそのまま再現する:

1. エキスパートのロジットを S seed で平均する（OOF側もtest側も同じseed集合）
   → メタの学習入力とtest入力のノイズ水準が揃う（errors-in-variables の是正）
2. 平均済みスタックの上に **メタを1本**張る（1se則でalpha選択）
3. 閾値は `cbpe_platt`（AのOOFで1変数Platt校正 → testの未ラベルスコア上でCBPE
   推定F1を最大化 → 生スコアへ逆変換）。testラベルは一切使わない

seed は **性能で選ばず添字で 0,1,2 に固定**する。seed42 は単seedのOOF F1@kが
9seed中の最高タイ（0.8165 対 平均0.8096）で、含めると平均が上振れて再現しない。

実行例:
  .venv/bin/python exp/make_submission_noise_matched_cbpe.py
  .venv/bin/python exp/make_submission_noise_matched_cbpe.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))

from ensemble_experts import expert_names  # noqa: E402
from holdout_check import _best_th, _cbpe_platt_threshold  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha_1se, to_logit  # noqa: E402

TAG = "concatemb_crosslr_lin0.03_h2h7both4v2eq"
SEEDS = (0, 1, 2, 3, 4)          # 先頭から --n-seeds 本だけ使う（性能で選ばない）
ALPHA_SEED = 42                  # alpha選択の内側CV用。本番の慣例に合わせて固定
DEFAULT_OUTPUT = ROOT / "submission/submission_h2h7both4_1se_s3_cbpeplatt.csv"


def build(n_seeds: int):
    names = expert_names(cross=True, linear=True)
    seeds = list(SEEDS[:n_seeds])
    oof_stacks, test_stacks = [], []
    for seed in seeds:
        cache = np.load(ROOT / f"exp/_experts_seed{seed}_{TAG}.npz")
        y = cache["y"].astype(int)
        oof_stacks.append(to_logit(np.column_stack([cache[f"oof_{n}"] for n in names])))
        test_stacks.append(to_logit(np.column_stack([cache[f"te_{n}"] for n in names])))

    Z_oof = np.mean(oof_stacks, axis=0)
    Z_test = np.mean(test_stacks, axis=0)

    alpha = select_alpha_1se(Z_oof, y, ALPHA_SEED)
    weight, intercept = fit_meta(Z_oof, y, alpha=alpha)
    p_oof = apply_meta(Z_oof, weight, intercept)
    p_test = apply_meta(Z_test, weight, intercept)
    return y, p_oof, p_test, weight, alpha, seeds, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    y, p_oof, p_test, weight, alpha, seeds, names = build(args.n_seeds)
    th = _cbpe_platt_threshold(p_oof, y, p_test)
    label = (p_test >= th).astype(int)

    print(f"S={len(seeds)} seeds={seeds}  alpha={alpha:g}")
    print("メタ重み: " + "  ".join(f"{n}={w:.3f}" for n, w in zip(names, weight)))
    print(f"\ncbpe_platt 閾値 = {th:.6f}")
    print(f"  test予測正例 {int(label.sum())} / {len(label)} ({label.mean():.2%})")
    print(f"  参考: OOF最適閾値 oof_best = {_best_th(y, p_oof):.4f}"
          f" / そのOOF正例率 {(p_oof >= _best_th(y, p_oof)).mean():.2%}")
    print(f"  参考: OOF実正例率 {y.mean():.2%}  test予測平均 {p_test.mean():.2%}")

    test_ids = pd.read_csv(ROOT / "data/test.csv")["企業ID"].to_numpy()
    out = pd.DataFrame({0: test_ids, 1: label})
    sample = pd.read_csv(ROOT / "data/sample_submit.csv", header=None)
    assert out.shape == sample.shape, f"形が違う: {out.shape} vs {sample.shape}"
    assert (out[0].to_numpy() == sample[0].to_numpy()).all(), "ID列がsampleと違う"
    out.to_csv(args.output, index=False, header=False)
    print(f"\n保存: {args.output}")

    for other in ("submission/submission_h2h7both4_1se_slot1_halff1.csv",
                  "submission/submission_h2h7both4_1se_two_slot_emax_hedge.csv"):
        path = ROOT / other
        if not path.exists():
            continue
        prev = pd.read_csv(path, header=None)
        up = int(((prev[1] == 0) & (out[1].to_numpy() == 1)).sum())
        down = int(((prev[1] == 1) & (out[1].to_numpy() == 0)).sum())
        print(f"  vs {Path(other).name}: 差 {up + down}行 (0→1:{up} / 1→0:{down})")


if __name__ == "__main__":
    main()
