"""Privateで高い方が採用される提出2枠を、E[max(F1)]で使い分ける。

枠1は ``half_f1`` 閾値の提出。枠2だけを、OOFでPlatt較正したtest確率から
生成するposterior predictive simulationで選ぶ。Publicラベルは使わない。

``--alpha-rule 1se`` を渡すと、両方の枠の土台になる合成そのものを
1-SE則のメタに差し替える（documents/meta_native_and_alpha_selection_20260829.md）。
土台のランキングが変われば half_f1 閾値も E[max] 曲線も引き直しになるので、
枠1・枠2を同時に作り直す。

**注意**: alpha-rule をまたいだ E[max] の数値は直接比較できない。この
シミュレーションは「その土台自身の較正済み確率」を真値として期待値を取るので、
別の土台どうしでは真値の定義が違う。ランキングの優劣は AP/AUC のペア比較
（11seedで ΔAP +0.0030, 9/11）で判断すること。

実行例:
  .venv/bin/python exp/make_submission_two_slot_emax.py
  .venv/bin/python exp/make_submission_two_slot_emax.py --alpha-rule 1se
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))

from ensemble_experts import _scores, blend_oof, expert_names  # noqa: E402
from meta_blend import to_logit  # noqa: E402


CACHE = ROOT / "exp/_experts_seed42_concatemb_crosslr_lin0.03_h2h7both4v2eq.npz"
MAIN_SUBMISSION = ROOT / (
    "submission/submission_ensemble_experts_seed42_aauto_concatemb_"
    "crosslr_lin003_h2h7both4v2eq_th04152542372881356.csv"
)
DEFAULT_OUTPUT = ROOT / "submission/submission_h2h7both4_two_slot_emax_hedge_top189.csv"
DEFAULT_CURVE = ROOT / "exp/two_slot_emax_curve_seed20260829.csv"
RNG_SEED = 20260829
PRIVATE_RATE = 0.70
BOOTSTRAPS = 500
DRAWS_PER_BOOTSTRAP = 100
PLATEAU_TOL = 1e-4


def _load_predictions(alpha_rule="auto"):
    names = expert_names(cross=True, linear=True)
    expected = (
        "E0_anchor", "E1_finance", "E2_survey", "E3_dx_text",
        "E4_org_text", "E6_manual", "E7_cross", "E0b_linear",
    )
    assert names == expected, f"本番8 Expert構成が変わった: {names}"
    cache = np.load(CACHE)
    y = cache["y"].astype(int)
    oof = {name: cache[f"oof_{name}"] for name in names}
    test = {name: cache[f"te_{name}"] for name in names}
    p_oof, p_test, _, _ = blend_oof(
        y, oof, test, seed=42, alpha=None, experts=names,
        alpha_rule=alpha_rule,
    )
    return y, p_oof, p_test


def _simulate_curve(y, p_oof, p_test, main_k):
    """較正パラメータとPrivate標本の不確実性を同時に積分する。"""
    rng = np.random.default_rng(RNG_SEED)
    order = np.argsort(-p_test)
    x_oof = to_logit(p_oof).reshape(-1, 1)
    x_test = to_logit(p_test).reshape(-1, 1)
    ks = np.arange(max(1, main_k - 60), min(len(p_test), main_k + 60) + 1)
    single_sum = np.zeros(len(ks))
    max_sum = np.zeros(len(ks))
    win_sum = np.zeros(len(ks))
    n_draws = 0

    for _ in range(BOOTSTRAPS):
        boot = rng.integers(0, len(y), len(y))
        calibrator = LogisticRegression(C=1e6, max_iter=3000)
        calibrator.fit(x_oof[boot], y[boot])
        prob = calibrator.predict_proba(x_test)[:, 1][order]

        n = DRAWS_PER_BOOTSTRAP
        private = rng.random((n, len(prob))) < PRIVATE_RATE
        label = rng.random((n, len(prob))) < prob
        true_positive_prefix = np.cumsum(private * label, axis=1)
        predicted_prefix = np.cumsum(private, axis=1)
        actual_positive = (private * label).sum(axis=1)

        main_tp = true_positive_prefix[:, main_k - 1]
        main_f1 = 2 * main_tp / np.maximum(
            predicted_prefix[:, main_k - 1] + actual_positive, 1,
        )
        candidate_f1 = np.stack([
            2 * true_positive_prefix[:, k - 1] / np.maximum(
                predicted_prefix[:, k - 1] + actual_positive, 1,
            )
            for k in ks
        ], axis=1)
        pair_f1 = np.maximum(candidate_f1, main_f1[:, None])
        single_sum += candidate_f1.sum(axis=0)
        max_sum += pair_f1.sum(axis=0)
        win_sum += (candidate_f1 > main_f1[:, None]).sum(axis=0)
        n_draws += n

    curve = pd.DataFrame({
        "positive_count": ks,
        "expected_single_f1": single_sum / n_draws,
        "expected_pair_max_f1": max_sum / n_draws,
        "candidate_win_rate": win_sum / n_draws,
    })
    main_expected = float(
        curve.loc[curve["positive_count"] == main_k, "expected_single_f1"].iloc[0]
    )
    curve["expected_pair_gain"] = curve["expected_pair_max_f1"] - main_expected

    # Monte Carloの1行差に合わせ込まない。最大値から1e-4以内を同率とし、
    # その中央のKを採る。固定seedでは189件になり、oof_bestの正例率転写とも一致する。
    peak = float(curve["expected_pair_max_f1"].max())
    plateau = curve[curve["expected_pair_max_f1"] >= peak - PLATEAU_TOL]
    selected_k = int(np.median(plateau["positive_count"].to_numpy()))
    return curve, selected_k


def run(output: Path, curve_path: Path, overwrite: bool = False,
        alpha_rule: str = "auto", main_output: Path | None = None):
    for path in (output, curve_path, main_output):
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(f"既存ファイルは上書きしない: {path}")

    y, p_oof, p_test = _load_predictions(alpha_rule)
    score = _scores(y, p_oof)
    main_threshold = float(score["f1"] / 2.0)
    main_label = (p_test >= main_threshold).astype(int)
    main_k = int(main_label.sum())
    if alpha_rule == "auto":
        assert main_k == 207, f"本命の正例数が変わった: {main_k}"

    test = pd.read_csv(ROOT / "data/test.csv")
    sample = pd.read_csv(
        ROOT / "data/sample_submit.csv",
        header=None,
        names=["企業ID", "購入フラグ"],
    )
    main_frame = pd.DataFrame({"企業ID": test["企業ID"], "pred": main_label})
    main_ordered = sample[["企業ID"]].merge(main_frame, on="企業ID", how="left")
    if alpha_rule == "auto":
        # 既定の土台では、既に提出済みの枠1と1行たりとも違わないことを確認する。
        existing = pd.read_csv(
            MAIN_SUBMISSION, header=None, names=["企業ID", "購入フラグ"],
        )
        assert np.array_equal(existing["企業ID"], sample["企業ID"])
        assert np.array_equal(existing["購入フラグ"], main_ordered["pred"]), (
            "再計算した本命が既存提出と一致しない"
        )
    if main_output is not None:
        main_output.parent.mkdir(parents=True, exist_ok=True)
        main_ordered.assign(pred=main_ordered["pred"].astype(int)).to_csv(
            main_output, index=False, header=False, lineterminator="\n",
        )

    curve, selected_k = _simulate_curve(y, p_oof, p_test, main_k)
    order = np.argsort(-p_test)
    hedge_label = np.zeros(len(p_test), dtype=int)
    hedge_label[order[:selected_k]] = 1
    hedge_frame = pd.DataFrame({"企業ID": test["企業ID"], "pred": hedge_label})
    submission = sample[["企業ID"]].merge(hedge_frame, on="企業ID", how="left")
    assert submission.shape == (800, 2)
    assert submission["pred"].notna().all()
    assert set(submission["pred"].unique()) <= {0, 1}
    assert int(submission["pred"].sum()) == selected_k

    output.parent.mkdir(parents=True, exist_ok=True)
    submission.assign(pred=submission["pred"].astype(int)).to_csv(
        output, index=False, header=False, lineterminator="\n",
    )
    curve.to_csv(curve_path, index=False, lineterminator="\n")

    row = curve.loc[curve["positive_count"] == selected_k].iloc[0]
    removed = test.loc[(main_label == 1) & (hedge_label == 0), "企業ID"].tolist()
    added = test.loc[(main_label == 0) & (hedge_label == 1), "企業ID"].tolist()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"土台: alpha-rule={alpha_rule}")
    print(f"枠1: half_f1 th={main_threshold:.12f}, 正例={main_k}"
          + (f" -> {main_output}" if main_output is not None else ""))
    print(
        f"E[max]ヘッジ: 正例={selected_k}, "
        f"単体期待F1={row.expected_single_f1:.6f}, "
        f"二枠期待F1={row.expected_pair_max_f1:.6f}, "
        f"期待増分={row.expected_pair_gain:+.6f}, "
        f"勝率={row.candidate_win_rate:.3f}"
    )
    print(f"本命との差分: 1→0={len(removed)}, 0→1={len(added)}")
    print(f"1→0 IDs: {removed}")
    print(f"保存: {output}")
    print(f"曲線: {curve_path}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--alpha-rule", default="auto", choices=["auto", "1se"])
    parser.add_argument("--main-output", type=Path, default=None,
                        help="枠1も書き出す場合の出力先")
    args = parser.parse_args()
    run(args.output, args.curve, args.overwrite, args.alpha_rule,
        args.main_output)
