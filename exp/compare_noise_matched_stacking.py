"""施策1: ノイズ整合スタッキング（repeated cross-fitting）の strict A/B holdout。

## 何を測るか

本番では **メタの入力が学習時とtest時で別物**になっている。

- エキスパートのOOF予測 = **1モデル/行**（そのfoldで学習しなかった1本）
- エキスパートのtest予測 = **5fold平均**（`ensemble_experts.py` の `te[name]` は
  foldごとに積んで平均する）

同じ列のseed間sd(logit)を比べると OOF/test 比は E0 3.7x / E0b 6.6x / E7 16x /
E2 23.6x / E4 30x。メタは**ノイズの多い入力で学習してノイズの少ない入力に適用**して
おり、errors-in-variables で係数が減衰する。減衰量が列ごとに違うので重み配分自体が歪む
（9seed平均OOFで張り直すと正規化配分の L1 変化 0.102）。

候補は「A の中で S seed 分の cross-fitting を繰り返し、OOF も B予測も同じ S seed で
平均してからメタを張る」。入力のノイズ水準が両側で揃う。

## 既存ハーネスとの違い（重要）

`holdout_check.one_rep` は B側を **A全体で1回学習した単一モデル**で予測する。これは
本番（5fold平均）のノイズ構造を再現していないので、この仮説の検証には使えない。
本スクリプトは baseline 側も **5fold平均**で B を予測する＝本番と同じ構成にする。

baseline は候補の j=0 成分そのものなので、A/B分割・fold分割・alpha選択まで完全に
共有した**入れ子のペア比較**になる。

## 判定（実行前に固定、`exp/decision.py` の既定に従う）

n=11 なら ΔAUC正 8/11以上（検出）、ΔAP正 9/11以上かつ平均>0（F1関連）、
平均ΔF1 ≥ -0.002（ガードレール）、平均ΔAP ≥ 0.005 で「明確」。

主指標は **k を揃えた ΔF1**。閾値を各アームに選ばせると作用点の比較が混ざる
（exp036 が 236→223 の k移動を「順位器の劣化」と読み違えた原因）。
各アーム自前の閾値でのF1も副指標として併記する。

`documents/ab_holdout_false_positive.md` の偽陽性パターン検出のため、A側APも記録して
B側APと符号が一致するかを見る。

実行例:
  OMP_NUM_THREADS=4 .venv/bin/python exp/compare_noise_matched_stacking.py \
      --n-reps 11 --n-seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))

from compare_meta_losses import CURRENT_LINEAR_C, EXPERTS, _inputs  # noqa: E402
from holdout_check import _best_th, _thresholds, fit_predict_experts  # noqa: E402
from meta_blend import apply_meta, fit_meta, select_alpha_1se, to_logit  # noqa: E402

DEFAULT_OUTPUT = ROOT / "exp/holdout_noise_matched_stacking_11rep.csv"
N_SPLITS = 5
TEST_SIZE = 0.35
SEED_STRIDE = 1000  # j番目の繰り返しに使う fold seed = rep + SEED_STRIDE * j

# 閾値則。すべて A のラベル（と B の未ラベルスコア）だけで決まる。
# cbpe_platt は保留中の候補（20repで +0.0122・分散46-77%減、本番切替はPublic 1ドローで保留）。
RULES = ("oof_best", "half_f1", "cbpe_platt")


def _cross_fit_once(X, y, text, org, e4, columns, A, B, seed):
    """A の中で1回 cross-fitting し、(A側OOF, B側5fold平均) を返す。

    B は fold ごとに同じモデルで予測して平均する＝本番 `ensemble_experts` と同じ。
    """
    yA = y[A]
    oof = {n: np.zeros(len(A)) for n in EXPERTS}
    parts = {n: [] for n in EXPERTS}
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                            random_state=seed).split(A, yA)
    for inner_train, inner_valid in folds:
        target = np.concatenate([A[inner_valid], B])
        pred = fit_predict_experts(
            X, y, text, org, A[inner_train], target, seed, columns,
            e4=e4, e7_model="lr", linear_c=CURRENT_LINEAR_C, strict_lgbm=True,
        )
        cut = len(inner_valid)
        for name in EXPERTS:
            oof[name][inner_valid] = pred[name][:cut]
            parts[name].append(pred[name][cut:])
    te = {n: np.mean(parts[n], axis=0) for n in EXPERTS}
    return oof, te


def _meta_arm(stack_A, stack_B, yA, seed):
    """A側スタックでalphaと重みを決めて、両側の合成確率を返す。"""
    alpha = select_alpha_1se(stack_A, yA, seed)
    weight, intercept = fit_meta(stack_A, yA, alpha=alpha)
    return (apply_meta(stack_A, weight, intercept),
            apply_meta(stack_B, weight, intercept), alpha, weight)


def _f1_at_k(y, p, k):
    label = np.zeros(len(p), dtype=int)
    label[np.argsort(-p)[:k]] = 1
    return float(f1_score(y, label))


def one_rep(X, y, text, org, e4, columns, rep, n_seeds):
    indices = np.arange(len(y))
    A, B = train_test_split(indices, test_size=TEST_SIZE, stratify=y,
                            random_state=rep)
    yA, yB = y[A], y[B]

    logit_oof, logit_te = [], []
    for j in range(n_seeds):
        oof, te = _cross_fit_once(X, y, text, org, e4, columns, A, B,
                                  seed=rep + SEED_STRIDE * j)
        logit_oof.append(to_logit(np.column_stack([oof[n] for n in EXPERTS])))
        logit_te.append(to_logit(np.column_stack([te[n] for n in EXPERTS])))
        print(f"    rep{rep} seed{j + 1}/{n_seeds} 完了", flush=True)

    # baseline は j=0 成分そのもの。S を増やした版を同じ土台の上で比べる。
    arms = {1: (logit_oof[0], logit_te[0])}
    for s in sorted({3, n_seeds} & set(range(2, n_seeds + 1))):
        arms[s] = (np.mean(logit_oof[:s], axis=0), np.mean(logit_te[:s], axis=0))

    fitted = {s: _meta_arm(za, zb, yA, rep) for s, (za, zb) in arms.items()}

    # k は baseline が A で選んだ閾値の正例率から決め、全アームで共有する。
    pA_base = fitted[1][0]
    th_base = _best_th(yA, pA_base)
    k_shared = int(round(float((pA_base >= th_base).mean()) * len(B)))

    rows = []
    for s, (pA, pB, alpha, weight) in fitted.items():
        ths = _thresholds(pA, yA, pB, folds=None, rules=RULES)
        row = {
            "rep": rep, "n_seeds": s,
            "auc": roc_auc_score(yB, pB),
            "ap": average_precision_score(yB, pB),
            "f1_matched_k": _f1_at_k(yB, pB, k_shared),
            "train_ap": average_precision_score(yA, pA),
            "train_auc": roc_auc_score(yA, pA),
            "k_shared": k_shared, "alpha": alpha,
            "n_A": len(A), "n_B": len(B),
            **{f"w_{n}": float(w) for n, w in zip(EXPERTS, weight)},
        }
        for rule, th in ths.items():
            row[f"f1_{rule}"] = float(f1_score(yB, (pB >= th).astype(int)))
            row[f"th_{rule}"] = float(th)
            row[f"pos_{rule}"] = int((pB >= th).sum())
        rows.append(row)
    # アーム間で top-k 集合がどれだけ入れ替わるか（効果量の目安）
    top_base = set(np.argsort(-fitted[1][1])[:k_shared])
    for row in rows:
        top_s = set(np.argsort(-fitted[row["n_seeds"]][1])[:k_shared])
        row["topk_overlap_vs_base"] = len(top_base & top_s) / k_shared
    # 後から別の閾値則を試せるように確率そのものを残す（再実行80分を避ける）
    probs = {f"{s}_{side}": arr for s, (pA, pB, _, _) in fitted.items()
             for side, arr in (("pA", pA), ("pB", pB))}
    probs.update({"yA": yA, "yB": yB, "A": A, "B": B})
    return rows, probs


def _wtl(delta):
    """勝/分/負と、負がゼロなら両側符号検定のp値。"""
    delta = delta.round(9)
    win, tie, loss = int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())
    note = f"  符号検定 p={2 ** -win * 2:.3f}" if loss == 0 and win > 0 else ""
    return win, tie, loss, note


def _paired_report(frame, n_seeds):
    base = frame[frame.n_seeds == 1].set_index("rep")
    f1_keys = [("f1_matched_k", "F1@k固定")] + [
        (f"f1_{r}", f"F1@{r}") for r in RULES]

    print("\n=== 2x2グリッド: 絶対値 (rep平均, B側) ===")
    print(f"  {'閾値則':16s} " + " ".join(f"S={s:<9d}" for s in sorted(frame.n_seeds.unique())))
    for key, label in f1_keys:
        cells = " ".join(f"{frame[frame.n_seeds == s][key].mean():<11.4f}"
                         for s in sorted(frame.n_seeds.unique()))
        print(f"  {label:16s} {cells}")
    print(f"\n  {'閾値則':16s} " + " ".join(f"最悪rep S={s:<3d}" for s in sorted(frame.n_seeds.unique())))
    for key, label in f1_keys:
        cells = " ".join(f"{frame[frame.n_seeds == s][key].min():<11.4f}"
                         for s in sorted(frame.n_seeds.unique()))
        print(f"  {label:16s} {cells}")

    print("\n=== Q1: S=1 の上で cbpe_platt は oof_best に勝つか ===")
    for rule in RULES:
        if rule == "oof_best":
            continue
        delta = base[f"f1_{rule}"] - base["f1_oof_best"]
        win, tie, loss, note = _wtl(delta)
        print(f"  {rule:12s} {base['f1_oof_best'].mean():.4f} -> {base[f'f1_{rule}'].mean():.4f}"
              f"   Δ{delta.mean():+.4f}   勝{win} 分{tie} 負{loss}"
              f"   最悪rep {delta.min():+.4f}"
              f"   rep間sd {base['f1_oof_best'].std():.4f}->{base[f'f1_{rule}'].std():.4f}{note}")

    print("\n=== Q2: ノイズ整合の利得は閾値則ごとにどこまで届くか ===")
    for s in sorted(frame.n_seeds.unique()):
        if s == 1:
            continue
        cand = frame[frame.n_seeds == s].set_index("rep")
        print(f"\n--- S = {s} seed 平均 (vs S=1) ---")
        for key, label in (("auc", "AUC"), ("ap", "AP"), ("train_ap", "A側AP(対照)")) + tuple(f1_keys):
            delta = cand[key] - base[key]
            win, tie, loss, note = _wtl(delta)
            print(f"  {label:14s} {base[key].mean():.4f} -> {cand[key].mean():.4f}"
                  f"   Δ{delta.mean():+.4f}   勝{win} 分{tie} 負{loss}"
                  f"   最悪rep {delta.min():+.4f}{note}")
        print(f"  {'top-k入替':14s} {((1 - cand.topk_overlap_vs_base) * cand.k_shared).mean():.1f}行/rep")

    print("\n=== 判定 (exp/decision.py の既定, 主指標=F1@k固定) ===")
    from decision import verdict
    cand = frame[frame.n_seeds == n_seeds].set_index("rep")
    base_df = base.rename(columns={"f1_matched_k": "f1"})[["auc", "ap", "f1"]]
    cand_df = cand.rename(columns={"f1_matched_k": "f1"})[["auc", "ap", "f1"]]
    v, _ = verdict(base_df.reset_index(drop=True), cand_df.reset_index(drop=True))
    print(f"  S={n_seeds} => {v}")

    print("\n=== 予測正例数 (rep平均, B=260行) ===")
    for rule in RULES:
        cells = " ".join(f"S={s}:{frame[frame.n_seeds == s][f'pos_{rule}'].mean():5.1f}"
                         for s in sorted(frame.n_seeds.unique()))
        print(f"  {rule:12s} {cells}")

    print("\n=== メタ重み (rep平均) ===")
    wcols = [c for c in frame.columns if c.startswith("w_")]
    print(frame.groupby("n_seeds")[wcols].mean().T.to_string(
        float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=11)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    X, y, text, org, e4, columns = _inputs()
    n_B = int(np.ceil(len(y) * TEST_SIZE))
    print(f"A={len(y) - n_B}行 / B={n_B}行 x {args.n_reps}rep, "
          f"S={args.n_seeds} seedのrepeated cross-fitting, "
          f"B側は両アームとも5fold平均（本番と同じ）")

    rows, saved = [], {}
    for rep in range(args.n_reps):
        print(f"  rep {rep + 1}/{args.n_reps} ...", flush=True)
        rep_rows, probs = one_rep(X, y, text, org, e4, columns, rep,
                                  args.n_seeds)
        rows.extend(rep_rows)
        saved.update({f"rep{rep}_{k}": v for k, v in probs.items()})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    npz_path = Path(args.output).with_suffix(".npz")
    np.savez_compressed(npz_path, **saved)
    print(f"保存: {args.output}\n保存: {npz_path}（確率そのもの。閾値則の追試は再実行不要）")
    _paired_report(frame, args.n_seeds)


if __name__ == "__main__":
    main()
