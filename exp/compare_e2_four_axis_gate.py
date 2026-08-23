"""E2を4軸だけで条件付けする、小自由度のcross-fitted probability gate。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_llm_make_or_buy import BASE_EXPERTS  # noqa: E402
from ensemble_experts import THS, _scores, blend_oof  # noqa: E402


CACHE = "exp/_experts_seed42_concatemb_crosslr_lin0.03_llmcore4e0onlyh2both4.npz"
E2 = "E2_survey"
AXES = ("アンケート１０", "internal_training_capability_score",
        "buy_openness_score", "implementation_stage_ordinal")
GATE_SEEDS = (42, 0, 1, 2, 3)
L2 = 0.10


def fit_gate(x, y, p_full, p_no):
    """g(x)*p_full + (1-g(x))*p_no のloglossを最小化する。"""
    eps = 1e-7

    def objective(theta):
        g = expit(theta[0] + x @ theta[1:])
        p = np.clip(p_no + g * (p_full - p_no), eps, 1 - eps)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss + 0.5 * L2 * np.dot(theta[1:], theta[1:])

    # mixture objectiveは非凸なので、同じモデルを決定的な多点初期化で解く。
    rng = np.random.default_rng(20260823)
    starts = [np.r_[v, np.zeros(x.shape[1])] for v in (0.0, 2.0, 5.0)]
    starts += [np.r_[2.0, rng.normal(0, 1.5, x.shape[1])] for _ in range(12)]
    results = [minimize(objective, start, method="L-BFGS-B") for start in starts]
    result = min((r for r in results if r.success), key=lambda r: r.fun)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def crossfit_gate(x, y, p_full, p_no, seed):
    out, gates = np.zeros(len(y)), np.zeros(len(y))
    coefs = []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in cv.split(x, y):
        imp = SimpleImputer(strategy="median").fit(x[tr])
        scaler = StandardScaler().fit(imp.transform(x[tr]))
        xtr = scaler.transform(imp.transform(x[tr]))
        xva = scaler.transform(imp.transform(x[va]))
        theta = fit_gate(xtr, y[tr], p_full[tr], p_no[tr])
        g = expit(theta[0] + xva @ theta[1:])
        gates[va] = g
        out[va] = p_no[va] + g * (p_full[va] - p_no[va])
        coefs.append(theta)
    return out, gates, np.asarray(coefs)


def main():
    raw = pd.read_csv("data/train.csv")
    llm = pd.read_csv("data/llm_make_or_buy/train_make_or_buy_features.csv")
    frame = raw[["企業ID", "アンケート１０"]].merge(
        llm[["企業ID", *AXES[1:]]], on="企業ID", how="left", validate="one_to_one")
    x = frame[list(AXES)].apply(pd.to_numeric, errors="coerce").to_numpy()

    d = np.load(CACHE)
    y = d["y"].astype(int)
    oof = {n: d[f"oof_{n}"] for n in BASE_EXPERTS}
    te = {n: d[f"te_{n}"] for n in BASE_EXPERTS}
    full, _, _, _ = blend_oof(y, oof, te, 42, experts=BASE_EXPERTS)
    without = tuple(n for n in BASE_EXPERTS if n != E2)
    no_e2, _, _, _ = blend_oof(y, oof, te, 42, experts=without)
    base_score = _scores(y, full)

    rows, fold_coefs = [], []
    for seed in GATE_SEEDS:
        pred, gate, coefs = crossfit_gate(x, y, full, no_e2, seed)
        score = _scores(y, pred)
        rows.append({"seed": seed, **score,
                     "d_auc": score["auc"] - base_score["auc"],
                     "d_ap": score["ap"] - base_score["ap"],
                     "d_f1": score["f1"] - base_score["f1"],
                     "gate_mean": gate.mean(), "gate_std": gate.std(),
                     "gate_min": gate.min(), "gate_max": gate.max()})
        fold_coefs.append(coefs)
        print(seed, rows[-1], flush=True)

    result = pd.DataFrame(rows)
    result.to_csv("exp/result_e2_four_axis_gate.csv", index=False)
    mean = result[["d_auc", "d_ap", "d_f1", "gate_mean", "gate_std"]].mean().to_dict()
    wins = {m: int((result[m] > 0).sum()) for m in ("d_auc", "d_ap", "d_f1")}
    coef = np.concatenate(fold_coefs)
    coef_summary = {"intercept": float(coef[:, 0].mean())}
    coef_summary.update({name: float(coef[:, i + 1].mean()) for i, name in enumerate(AXES)})
    payload = {"method": "four-axis probability gate", "axes": AXES, "l2": L2,
               "baseline": base_score, "mean": mean, "wins": wins,
               "mean_coefficients_standardized": coef_summary,
               "verdict": "PROMISING" if mean["d_ap"] > 0 and wins["d_ap"] >= 4 else "REJECT"}
    with open("exp/result_e2_four_axis_gate.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n", json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
